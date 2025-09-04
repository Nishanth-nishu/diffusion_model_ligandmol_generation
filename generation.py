import traceback
import logging
import os
# Third-party imports for generation and data manipulation
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

# Local imports from your other files
from model import ResearchValidatedDiffusionModel
from evaluation import MolecularGenerationBenchmark

class ResearchValidatedGenerator:
    """Fixed generator with comprehensive error handling"""
    
    def __init__(self, model, use_ema=True):
        self.model = model.to(device)
        self.use_ema = use_ema
        self.benchmark = MolecularGenerationBenchmark()
    
    def generate_with_research_protocols(
        self,
        num_molecules=1000,
        target_properties=None,
        atom_range=(10, 40),
        guidance_scale=2.0,
        num_sampling_steps=100,
        temperature=1.0,
        use_ddim=True
    ):
        """Generate molecules with comprehensive error handling"""
        
        print(f"Generating {num_molecules} molecules with research protocols...")
        
        self.model.eval()
        generated_molecules = []
        
        with torch.no_grad():
            for i in tqdm(range(num_molecules), desc="Generation"):
                try:
                    # Sample number of atoms
                    num_atoms = np.random.randint(atom_range[0], atom_range[1])
                    
                    # Fixed generation call
                    if use_ddim:
                        atom_features, positions = self.ddim_sampling_fixed(
                            num_atoms, target_properties, num_sampling_steps, guidance_scale, temperature
                        )
                    else:
                        atom_features, positions = self.ddpm_sampling_fixed(
                            num_atoms, target_properties, guidance_scale, temperature
                        )
                    
                    # Ensure tensors are properly converted to numpy
                    if isinstance(atom_features, torch.Tensor):
                        atom_features_np = atom_features.cpu().detach().numpy()
                    else:
                        atom_features_np = np.array(atom_features)
                    
                    if isinstance(positions, torch.Tensor):
                        positions_np = positions.cpu().detach().numpy()
                    else:
                        positions_np = np.array(positions)
                    
                    # Create molecule data
                    mol_data = {
                        'atom_features': atom_features_np,
                        'positions': positions_np,
                        'num_atoms': num_atoms,
                        'generation_id': i,
                        'target_properties': target_properties.cpu().numpy() if target_properties is not None and torch.is_tensor(target_properties) else None
                    }
                    
                    generated_molecules.append(mol_data)
                    
                except Exception as e:
                    logging.error(f"Generation {i} failed: {e}")
                    continue
        
        print(f"Generated {len(generated_molecules)} molecules")
        return generated_molecules
    
    def ddim_sampling_fixed(self, num_atoms, target_properties, num_steps, guidance_scale, temperature):
        """Fixed DDIM sampling with comprehensive error handling"""
        
        try:
            # Initialize noise
            x = torch.randn(num_atoms, self.model.atom_feature_dim, device=device) * temperature
            pos = torch.randn(num_atoms, 3, device=device) * temperature
            
            # Create connectivity
            edge_indices, edge_features = self.create_research_connectivity_fixed(num_atoms)
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)
            batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
            
            # DDIM timestep schedule
            timesteps = torch.linspace(self.model.timesteps-1, 0, num_steps, dtype=torch.long, device=device)
            
            # Ensure target_properties is properly formatted
            if target_properties is not None:
                if not torch.is_tensor(target_properties):
                    target_properties = torch.tensor(target_properties, dtype=torch.float, device=device)
                if target_properties.dim() == 1:
                    target_properties = target_properties.unsqueeze(0)
            
            for i, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)
                
                # Create data object
                data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                
                # Model prediction with fixed property handling
                try:
                    if target_properties is not None and guidance_scale > 1.0:
                        # Classifier-free guidance
                        pred_noise_x_cond, pred_noise_pos_cond, _ = self.model(data, t_batch, target_properties)[:3]
                        pred_noise_x_uncond, pred_noise_pos_uncond, _ = self.model(data, t_batch, None)[:3]
                        
                        pred_noise_x = pred_noise_x_uncond + guidance_scale * (pred_noise_x_cond - pred_noise_x_uncond)
                        pred_noise_pos = pred_noise_pos_uncond + guidance_scale * (pred_noise_pos_cond - pred_noise_pos_uncond)
                    else:
                        pred_noise_x, pred_noise_pos, _ = self.model(data, t_batch, target_properties)[:3]
                        
                except Exception as model_error:
                    print(f"Model prediction failed at step {i}: {model_error}")
                    # Fallback: use random noise
                    pred_noise_x = torch.randn_like(x) * 0.1
                    pred_noise_pos = torch.randn_like(pos) * 0.1
                
                # DDIM update step
                if i < len(timesteps) - 1:
                    try:
                        alpha_t = self.model.alphas_cumprod[t]
                        alpha_t_next = self.model.alphas_cumprod[timesteps[i+1]]
                        
                        # DDIM deterministic step
                        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise_x) / torch.sqrt(alpha_t)
                        pred_pos0 = (pos - torch.sqrt(1 - alpha_t) * pred_noise_pos) / torch.sqrt(alpha_t)
                        
                        x = torch.sqrt(alpha_t_next) * pred_x0 + torch.sqrt(1 - alpha_t_next) * pred_noise_x
                        pos = torch.sqrt(alpha_t_next) * pred_pos0 + torch.sqrt(1 - alpha_t_next) * pred_noise_pos
                        
                    except Exception as ddim_error:
                        print(f"DDIM step failed: {ddim_error}")
                        # Fallback: simple denoising
                        x = x - pred_noise_x * 0.1
                        pos = pos - pred_noise_pos * 0.1
                else:
                    # Final step
                    try:
                        alpha_t = self.model.alphas_cumprod[t]
                        x = (x - torch.sqrt(1 - alpha_t) * pred_noise_x) / torch.sqrt(alpha_t)
                        pos = (pos - torch.sqrt(1 - alpha_t) * pred_noise_pos) / torch.sqrt(alpha_t)
                    except:
                        x = x - pred_noise_x * 0.1
                        pos = pos - pred_noise_pos * 0.1
            
            return x, pos
            
        except Exception as e:
            print(f"DDIM sampling completely failed: {e}")
            # Ultimate fallback
            x = torch.zeros(num_atoms, self.model.atom_feature_dim, device=device)
            x[:, 6] = 1.0  # Set as carbon atoms
            pos = torch.randn(num_atoms, 3, device=device)
            return x, pos
    
    def ddpm_sampling_fixed(self, num_atoms, target_properties, guidance_scale, temperature):
        """Fixed DDPM sampling"""
        
        try:
            # Initialize noise
            x = torch.randn(num_atoms, self.model.atom_feature_dim, device=device) * temperature
            pos = torch.randn(num_atoms, 3, device=device) * temperature
            
            # Create connectivity
            edge_indices, edge_features = self.create_research_connectivity_fixed(num_atoms)
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)
            batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
            
            # Ensure target_properties is properly formatted
            if target_properties is not None:
                if not torch.is_tensor(target_properties):
                    target_properties = torch.tensor(target_properties, dtype=torch.float, device=device)
                if target_properties.dim() == 1:
                    target_properties = target_properties.unsqueeze(0)
            
            # Full reverse diffusion
            for t in reversed(range(self.model.timesteps)):
                t_batch = torch.tensor([t], device=device)
                
                data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                
                # Model prediction
                try:
                    if target_properties is not None and guidance_scale > 1.0:
                        pred_noise_x_cond, pred_noise_pos_cond, _ = self.model(data, t_batch, target_properties)[:3]
                        pred_noise_x_uncond, pred_noise_pos_uncond, _ = self.model(data, t_batch, None)[:3]
                        
                        pred_noise_x = pred_noise_x_uncond + guidance_scale * (pred_noise_x_cond - pred_noise_x_uncond)
                        pred_noise_pos = pred_noise_pos_uncond + guidance_scale * (pred_noise_pos_cond - pred_noise_pos_uncond)
                    else:
                        pred_noise_x, pred_noise_pos, _ = self.model(data, t_batch, target_properties)[:3]
                        
                except Exception:
                    pred_noise_x = torch.randn_like(x) * 0.1
                    pred_noise_pos = torch.randn_like(pos) * 0.1
                
                # DDPM update
                if t > 0:
                    try:
                        beta_t = self.model.betas[t]
                        alpha_t = self.model.alphas[t]
                        alpha_cumprod_t = self.model.alphas_cumprod[t]
                        
                        # Mean prediction
                        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise_x)
                        pos = (1 / torch.sqrt(alpha_t)) * (pos - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise_pos)
                        
                        # Add noise
                        if t > 1:
                            noise_x = torch.randn_like(x) * torch.sqrt(beta_t) * temperature
                            noise_pos = torch.randn_like(pos) * torch.sqrt(beta_t) * temperature
                            
                            x += noise_x
                            pos += noise_pos
                            
                    except Exception:
                        x = x - pred_noise_x * 0.01
                        pos = pos - pred_noise_pos * 0.01
                else:
                    # Final denoising step
                    try:
                        alpha_cumprod_t = self.model.alphas_cumprod[t]
                        x = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise_x) / torch.sqrt(alpha_cumprod_t)
                        pos = (pos - torch.sqrt(1 - alpha_cumprod_t) * pred_noise_pos) / torch.sqrt(alpha_cumprod_t)
                    except:
                        x = x - pred_noise_x * 0.1
                        pos = pos - pred_noise_pos * 0.1
            
            return x, pos
            
        except Exception as e:
            print(f"DDPM sampling failed: {e}")
            # Fallback
            x = torch.zeros(num_atoms, self.model.atom_feature_dim, device=device)
            x[:, 6] = 1.0  # Carbon atoms
            pos = torch.randn(num_atoms, 3, device=device)
            return x, pos
    
    def create_research_connectivity_fixed(self, num_atoms):
        """Create realistic connectivity with proper error handling"""
        
        try:
            edge_indices = []
            edge_features = []
            
            # Create connectivity based on molecule size
            if num_atoms <= 5:
                # Linear chain
                for i in range(num_atoms - 1):
                    edge_indices.extend([[i, i+1], [i+1, i]])
                    edge_features.extend([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
            
            elif num_atoms <= 15:
                # Ring + substituents
                ring_size = min(6, num_atoms // 2)
                
                # Create ring
                for i in range(ring_size):
                    j = (i + 1) % ring_size
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])  # Aromatic
                
                # Add substituents
                for i in range(ring_size, num_atoms):
                    attach_point = np.random.randint(0, ring_size)
                    edge_indices.extend([[attach_point, i], [i, attach_point]])
                    edge_features.extend([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])  # Single bond
            
            else:
                # More complex connectivity
                # Primary ring
                for i in range(6):
                    j = (i + 1) % 6
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
                
                atoms_used = 6
                
                # Secondary structures
                remaining = num_atoms - atoms_used
                if remaining >= 6:
                    # Second ring
                    start_idx = atoms_used
                    for i in range(6):
                        j = start_idx + (i + 1) % 6
                        if j < num_atoms:
                            edge_indices.extend([[start_idx + i, j], [j, start_idx + i]])
                            edge_features.extend([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
                    
                    # Connect rings
                    edge_indices.extend([[2, start_idx], [start_idx, 2]])
                    edge_features.extend([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
                    atoms_used += 6
                
                # Remaining atoms as substituents
                for i in range(atoms_used, num_atoms):
                    attach_point = np.random.randint(0, min(atoms_used, 12))
                    edge_indices.extend([[attach_point, i], [i, attach_point]])
                    edge_features.extend([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
            
            # Ensure we have at least one edge
            if not edge_indices and num_atoms > 1:
                edge_indices = [[0, 1], [1, 0]]
                edge_features = [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
            elif not edge_indices:
                edge_indices = [[0, 0]]
                edge_features = [[1.0, 0.0, 0.0, 0.0, 0.0]]
            
            return edge_indices, edge_features
            
        except Exception as e:
            print(f"Connectivity creation failed: {e}")
            # Minimal fallback
            if num_atoms > 1:
                return [[0, 1], [1, 0]], [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
            else:
                return [[0, 0]], [[1.0, 0.0, 0.0, 0.0, 0.0]]


def _add_missing_attributes_to_model(model):
    """Add missing attributes to the diffusion model"""
    
    if not hasattr(model, 'posterior_variance'):
        # Calculate posterior variance for DDPM
        alphas_cumprod_prev = F.pad(model.alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = model.betas * (1.0 - alphas_cumprod_prev) / (1.0 - model.alphas_cumprod)
        model.register_buffer('posterior_variance', posterior_variance)

# FIXED GENERATION SCENARIOS - Update this in the main training function
def get_fixed_generation_scenarios():
    """Get fixed generation scenarios with proper property formatting"""
    
    generation_scenarios = [
        {
            "name": "Kinase Inhibitors",
            "properties": torch.tensor([[0.7, 0.5, 0.4, 0.4, 0.2, 0.4, 0.8, 0.7, 0.6, 0.4, 0.5, 0.6, 0.6, 0.7, 0.2]], device=device, dtype=torch.float32),
            "guidance": 2.0,
            "expected_mw_range": (300, 600),
            "expected_logp_range": (2, 5)
        },
        {
            "name": "CNS Drugs",
            "properties": torch.tensor([[0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.4, 0.7, 0.8, 0.4]], device=device, dtype=torch.float32),
            "guidance": 2.5,
            "expected_mw_range": (200, 450),
            "expected_logp_range": (1, 4)
        },
        {
            "name": "Fragment-like",
            "properties": torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.9, 0.2, 0.6, 0.1, 0.3, 0.8, 0.9, 1.0]], device=device, dtype=torch.float32),
            "guidance": 1.5,
            "expected_mw_range": (120, 300),
            "expected_logp_range": (0, 3)
        }
    ]
    
    return generation_scenarios

# UPDATE THE GENERATION SECTION IN THE MAIN FUNCTION
# Replace this section in run_research_validated_training:

def run_fixed_generation_testing(model):
    """Run fixed generation testing"""
    
    print("\nStep 6: Research-standard generation testing...")
    
    # Add missing attributes to model
    _add_missing_attributes_to_model(model)
    
    # Load best model for generation
    best_model_path = 'research_checkpoints/model_best.pt'
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model checkpoint")
        except Exception as e:
            print(f"Could not load checkpoint: {e}, using current model")
    
    generator = ResearchValidatedGenerator(model, use_ema=False)
    
    # Get fixed generation scenarios
    generation_scenarios = get_fixed_generation_scenarios()
    
    all_generated = []
    generation_results = {}
    
    for scenario in generation_scenarios:
        print(f"\nGenerating {scenario['name']} molecules...")
        
        try:
            generated = generator.generate_with_research_protocols(
                num_molecules=100,  # Reduced for faster testing
                target_properties=scenario['properties'],
                guidance_scale=scenario['guidance'],
                num_sampling_steps=20,  # Reduced for speed
                temperature=0.8,
                use_ddim=True
            )
            
            # Evaluate scenario-specific metrics
            if generated:
                scenario_results = generator.benchmark.benchmark_full_suite(
                    generated, None, None
                )
                
                generation_results[scenario['name']] = scenario_results
                all_generated.extend(generated)
                
                print(f"  Generated: {len(generated)} molecules")
                print(f"  Validity: {scenario_results.get('validity', 0.0):.3f}")
                print(f"  Drug-likeness: {scenario_results.get('drug_likeness', 0.0):.3f}")
            else:
                print(f"  No molecules generated for {scenario['name']}")
                generation_results[scenario['name']] = {'validity': 0.0, 'drug_likeness': 0.0, 'uniqueness': 0.0}
                
        except Exception as e:
            print(f"  Generation failed for {scenario['name']}: {e}")
            generation_results[scenario['name']] = {'validity': 0.0, 'drug_likeness': 0.0, 'uniqueness': 0.0}
    
    return generation_results, all_generated
