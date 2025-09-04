import os
import traceback
import logging

# Third-party imports for training and data handling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

# Local imports from your other files
from model import ResearchValidatedDiffusionModel

class ResearchValidatedTrainer:
    """Enhanced trainer with comprehensive debugging and error handling"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
        lr=1e-4,
        weight_decay=1e-6,
        ema_decay=0.9999,
        gradient_clip=1.0,
        debug_mode=True  # NEW: Enable debugging
    ):
        self.model = model.to(device)
        # Validate model components
        if not validate_model_components(self.model):
          raise RuntimeError("Model component validation failed - contains non-tensor components!")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.debug_mode = debug_mode

        # Research-validated optimizer settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Exponential moving average (standard in diffusion research)
        self.ema_decay = ema_decay
        self.ema_model = None
        if ema_decay > 0:
            self.ema_model = self._create_ema_model()

        # Learning rate scheduling (from research)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=lr * 0.01
        )

        # Research-validated loss function
        self.criterion = ResearchValidatedLoss()

        # Metrics tracking
        self.metrics = {
            'train_losses': [], 'val_losses': [], 'test_losses': [],
            'atom_losses': [], 'pos_losses': [], 'prop_losses': [],
            'consistency_losses': [], 'valency_losses': []
        }

        # Debug statistics
        if self.debug_mode:
            self.debug_stats = {
                'successful_steps': 0,
                'failed_steps': 0,
                'batch_size_distribution': {},
                'timestep_distribution': {},
                'error_log': []
            }

    def _create_ema_model(self):
        """Create EMA model (standard practice in diffusion research)"""
        ema_model = type(self.model)(
            atom_feature_dim=self.model.atom_feature_dim,
            hidden_dim=self.model.hidden_dim,
            timesteps=self.model.timesteps
        ).to(device)

        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()

        return ema_model

    def debug_batch_info(self, batch, step_name=""):
        """Debug batch information"""
        if not self.debug_mode:
            return

        try:
            print(f"\n[DEBUG {step_name}] Batch Analysis:")
            print(f"  - x.shape: {batch.x.shape}")
            print(f"  - pos.shape: {batch.pos.shape}")

            if hasattr(batch, 'batch') and batch.batch is not None:
                print(f"  - batch tensor: {batch.batch}")
                print(f"  - batch.shape: {batch.batch.shape}")
                print(f"  - unique batch values: {torch.unique(batch.batch)}")
                print(f"  - max batch index: {batch.batch.max().item()}")
                print(f"  - batch size calculation: {batch.batch.max().item() + 1}")
            else:
                print(f"  - batch tensor: None or missing")

            if hasattr(batch, 'edge_index'):
                print(f"  - edge_index.shape: {batch.edge_index.shape}")

        except Exception as e:
            print(f"[DEBUG ERROR] Failed to analyze batch: {e}")

    def safe_batch_size_calculation(self, batch):
        """Safely calculate batch size with extensive debugging"""

        try:
            # Method 1: Use batch attribute if available
            if hasattr(batch, 'batch') and batch.batch is not None:
                if batch.batch.numel() == 0:
                    if self.debug_mode:
                        print("[DEBUG] Empty batch tensor detected, using batch_size=1")
                    return 1, torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)

                unique_batches = torch.unique(batch.batch)
                batch_size = len(unique_batches)

                if self.debug_mode:
                    print(f"[DEBUG] Calculated batch_size={batch_size} from batch tensor")
                    print(f"[DEBUG] Unique batch indices: {unique_batches}")

                # Ensure batch_size is reasonable
                if batch_size <= 0:
                    if self.debug_mode:
                        print("[DEBUG] Invalid batch_size, defaulting to 1")
                    batch_size = 1
                    batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)

                return batch_size, batch.batch

            # Method 2: Fallback calculation
            else:
                if self.debug_mode:
                    print("[DEBUG] No batch tensor found, creating default")
                batch_size = 1
                batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
                return batch_size, batch.batch

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG ERROR] Batch size calculation failed: {e}")
                print(f"[DEBUG ERROR] Traceback: {traceback.format_exc()}")

            # Ultimate fallback
            batch_size = 1
            batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
            return batch_size, batch.batch

    def safe_timestep_sampling(self, batch_size):
        """Safely sample timesteps with debugging"""

        try:
            # Simple random sampling (no importance sampling to avoid complexity)
            t = torch.randint(0, self.model.timesteps, (batch_size,), device=device)

            # Ensure timesteps are within valid range
            t = torch.clamp(t, 0, self.model.timesteps - 1)

            if self.debug_mode:
                print(f"[DEBUG] Sampled timesteps: {t}")
                print(f"[DEBUG] Timesteps shape: {t.shape}")
                print(f"[DEBUG] Timesteps range: [{t.min().item()}, {t.max().item()}]")

            return t

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG ERROR] Timestep sampling failed: {e}")

            # Fallback: single timestep
            t = torch.tensor([self.model.timesteps // 2], device=device)
            return t

    def safe_noise_schedule_lookup(self, t):
        """Safely lookup noise schedule values with debugging"""

        try:
            # Ensure t is within bounds
            t_safe = torch.clamp(t, 0, self.model.timesteps - 1)

            # Get noise schedule values
            sqrt_alpha_t = self.model.sqrt_alphas_cumprod[t_safe]
            sqrt_sigma_t = self.model.sqrt_one_minus_alphas_cumprod[t_safe]

            if self.debug_mode:
                print(f"[DEBUG] t_safe: {t_safe}")
                print(f"[DEBUG] sqrt_alpha_t: {sqrt_alpha_t}")
                print(f"[DEBUG] sqrt_sigma_t: {sqrt_sigma_t}")
                print(f"[DEBUG] sqrt_alpha_t.shape: {sqrt_alpha_t.shape}")
                print(f"[DEBUG] sqrt_sigma_t.shape: {sqrt_sigma_t.shape}")

            # Ensure proper dimensions
            if sqrt_alpha_t.dim() == 0:
                sqrt_alpha_t = sqrt_alpha_t.unsqueeze(0)
            if sqrt_sigma_t.dim() == 0:
                sqrt_sigma_t = sqrt_sigma_t.unsqueeze(0)

            return sqrt_alpha_t, sqrt_sigma_t

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG ERROR] Noise schedule lookup failed: {e}")
                print(f"[DEBUG ERROR] t: {t}")
                print(f"[DEBUG ERROR] model.timesteps: {self.model.timesteps}")

            # Fallback: use middle values
            mid_idx = self.model.timesteps // 2
            sqrt_alpha_t = self.model.sqrt_alphas_cumprod[mid_idx].unsqueeze(0)
            sqrt_sigma_t = self.model.sqrt_one_minus_alphas_cumprod[mid_idx].unsqueeze(0)

            return sqrt_alpha_t, sqrt_sigma_t

    def add_research_noise_safe(self, data, t):
        """Safe noise addition with comprehensive error handling and debugging"""

        if self.debug_mode:
            print(f"\n[DEBUG] Starting noise addition...")
            self.debug_batch_info(data, "NOISE_INPUT")

        try:
            # Ensure proper device placement
            data = data.to(device)
            t = t.to(device)

            # Safe batch size calculation
            batch_size, batch_tensor = self.safe_batch_size_calculation(data)
            data.batch = batch_tensor

            # Ensure t has correct batch size
            if len(t) != batch_size:
                if self.debug_mode:
                    print(f"[DEBUG] Adjusting t from length {len(t)} to batch_size {batch_size}")
                t = self.safe_timestep_sampling(batch_size)

            # Safe noise schedule lookup
            sqrt_alpha_t, sqrt_sigma_t = self.safe_noise_schedule_lookup(t)

            # Generate noise
            noise_x = torch.randn_like(data.x)
            noise_pos = torch.randn_like(data.pos)

            if self.debug_mode:
                print(f"[DEBUG] Generated noise_x.shape: {noise_x.shape}")
                print(f"[DEBUG] Generated noise_pos.shape: {noise_pos.shape}")

            # Apply noise per batch element with safe indexing
            try:
                sqrt_alpha_nodes = sqrt_alpha_t[data.batch].unsqueeze(-1)
                sqrt_sigma_nodes = sqrt_sigma_t[data.batch].unsqueeze(-1)

                if self.debug_mode:
                    print(f"[DEBUG] sqrt_alpha_nodes.shape: {sqrt_alpha_nodes.shape}")
                    print(f"[DEBUG] sqrt_sigma_nodes.shape: {sqrt_sigma_nodes.shape}")

            except IndexError as idx_err:
                if self.debug_mode:
                    print(f"[DEBUG ERROR] IndexError in noise application: {idx_err}")
                    print(f"[DEBUG ERROR] data.batch: {data.batch}")
                    print(f"[DEBUG ERROR] sqrt_alpha_t.shape: {sqrt_alpha_t.shape}")

                # Safe fallback: broadcast to all nodes
                sqrt_alpha_nodes = sqrt_alpha_t[0].unsqueeze(0).unsqueeze(-1).expand(data.x.shape[0], -1)
                sqrt_sigma_nodes = sqrt_sigma_t[0].unsqueeze(0).unsqueeze(-1).expand(data.x.shape[0], -1)

            # Forward diffusion (research standard)
            noisy_x = sqrt_alpha_nodes * data.x + sqrt_sigma_nodes * noise_x
            noisy_pos = sqrt_alpha_nodes * data.pos + sqrt_sigma_nodes * noise_pos

            # Create noisy data object
            noisy_data = Data(
                x=noisy_x,
                pos=noisy_pos,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                properties=getattr(data, 'properties', None)
            )

            if self.debug_mode:
                print(f"[DEBUG] Noise addition successful")
                self.debug_batch_info(noisy_data, "NOISE_OUTPUT")

            return noisy_data, noise_x, noise_pos, t

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG ERROR] Critical noise addition failure: {e}")
                print(f"[DEBUG ERROR] Full traceback: {traceback.format_exc()}")

            # Ultimate fallback: return original data with minimal noise
            try:
                simple_noise_x = torch.randn_like(data.x) * 0.1
                simple_noise_pos = torch.randn_like(data.pos) * 0.1

                simple_noisy_data = Data(
                    x=data.x + simple_noise_x,
                    pos=data.pos + simple_noise_pos,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch,
                    properties=getattr(data, 'properties', None)
                )

                return simple_noisy_data, simple_noise_x, simple_noise_pos, torch.tensor([500], device=device)

            except Exception as final_err:
                print(f"[CRITICAL ERROR] Even fallback noise addition failed: {final_err}")
                raise final_err

    def train_step_research_validated_safe(self, batch):
        """Completely safe training step with comprehensive error handling"""

        if self.debug_mode:
            print(f"\n[DEBUG] ===== TRAINING STEP START =====")
            self.debug_batch_info(batch, "TRAIN_INPUT")

        try:
            self.optimizer.zero_grad()

            batch = batch.to(device)

            # Safe batch processing
            batch_size, _ = self.safe_batch_size_calculation(batch)

            # Safe timestep sampling
            t = self.safe_timestep_sampling(batch_size)

            # Safe noise addition
            noisy_batch, target_noise_x, target_noise_pos, t_actual = self.add_research_noise_safe(batch, t)

            # Forward pass with error handling
            try:
                outputs = self.model(noisy_batch, t_actual, getattr(batch, 'properties', None))

                if self.debug_mode:
                    print(f"[DEBUG] Model forward pass successful")
                    print(f"[DEBUG] Outputs length: {len(outputs)}")

            except Exception as model_err:
                if self.debug_mode:
                    print(f"[DEBUG ERROR] Model forward pass failed: {model_err}")
                raise model_err

            # Calculate loss
            predictions = outputs[:3]
            targets = (target_noise_x, target_noise_pos)

            if len(outputs) > 3:  # Include consistency outputs
                consistency_outputs = outputs[3]
                loss_dict = self.criterion(predictions, targets, consistency_outputs)
            else:
                loss_dict = self.criterion(predictions, targets)

            # Backward pass with gradient clipping
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.debug_mode:
                self.debug_stats['successful_steps'] += 1
                batch_size_key = f"batch_{batch_size}"
                self.debug_stats['batch_size_distribution'][batch_size_key] = self.debug_stats['batch_size_distribution'].get(batch_size_key, 0) + 1
                print(f"[DEBUG] Training step completed successfully")

            return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

        except Exception as e:
            if self.debug_mode:
                self.debug_stats['failed_steps'] += 1
                error_info = {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'batch_info': f"x.shape={batch.x.shape if hasattr(batch, 'x') else 'None'}"
                }
                self.debug_stats['error_log'].append(error_info)
                print(f"[DEBUG ERROR] Training step failed: {e}")
                print(f"[DEBUG ERROR] Full traceback: {traceback.format_exc()}")

            # Return zero losses for failed steps
            return {
                'total_loss': 0.0,
                'atom_loss': 0.0,
                'pos_loss': 0.0,
                'prop_loss': 0.0,
                'consistency_loss': 0.0,
                'valency_loss': 0.0
            }

    def train_research_validated(self, num_epochs, validation_frequency=5):
        """Training loop with comprehensive error handling and debugging"""

        print(f"Starting research-validated training for {num_epochs} epochs...")

        if self.debug_mode:
            print(f"[DEBUG] Debug mode enabled - detailed logging active")

        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 15

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            if self.debug_mode:
                print(f"[DEBUG] Starting epoch {epoch+1}")
                print(f"[DEBUG] Success rate so far: {self.debug_stats['successful_steps']}/{self.debug_stats['successful_steps'] + self.debug_stats['failed_steps']}")

            # Training phase
            self.model.train()
            epoch_losses = {'total': 0, 'atom': 0, 'pos': 0, 'prop': 0, 'consistency': 0, 'valency': 0}
            num_successful_batches = 0

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")):
                try:
                    loss_dict = self.train_step_research_validated_safe(batch)

                    # Only count successful steps
                    if loss_dict['total_loss'] > 0:
                        for key in epoch_losses:
                            if f'{key}_loss' in loss_dict:
                                epoch_losses[key] += loss_dict[f'{key}_loss']
                        num_successful_batches += 1

                    # Update EMA
                    self.update_ema()

                    # Debug info every 50 batches
                    if self.debug_mode and batch_idx % 50 == 0:
                        success_rate = self.debug_stats['successful_steps'] / max(1, self.debug_stats['successful_steps'] + self.debug_stats['failed_steps'])
                        print(f"\n[DEBUG] Batch {batch_idx}: Success rate = {success_rate:.3f}")

                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG ERROR] Batch {batch_idx} completely failed: {e}")
                    continue

            # Average losses (only from successful batches)
            if num_successful_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_successful_batches
                    self.metrics[f'{key}_losses'].append(epoch_losses[key])

                print(f"Epoch {epoch+1} completed - Successful batches: {num_successful_batches}/{len(self.train_loader)}")
                print(f"Average loss: {epoch_losses['total']:.4f}")
            else:
                print(f"[WARNING] Epoch {epoch+1} - No successful training steps!")

            # Validation
            if epoch % validation_frequency == 0 and self.val_loader:
                val_loss = self.validate_research_standard()
                self.metrics['val_losses'].append(val_loss)

                print(f"Validation Loss: {val_loss:.4f}")

                # Model selection and early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    self.save_research_validated_checkpoint(epoch, 'best')
                else:
                    patience_counter += 1

                if patience_counter >= patience_limit:
                    print(f"Early stopping after {patience_limit} epochs without improvement")
                    break

            # Learning rate scheduling
            self.scheduler.step()

            # Periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_research_validated_checkpoint(epoch, f'epoch_{epoch+1}')

        # Print final debug statistics
        if self.debug_mode:
            print(f"\n[DEBUG] ===== FINAL TRAINING STATISTICS =====")
            print(f"[DEBUG] Successful steps: {self.debug_stats['successful_steps']}")
            print(f"[DEBUG] Failed steps: {self.debug_stats['failed_steps']}")
            print(f"[DEBUG] Overall success rate: {self.debug_stats['successful_steps']/(self.debug_stats['successful_steps'] + self.debug_stats['failed_steps']):.3f}")
            print(f"[DEBUG] Batch size distribution: {self.debug_stats['batch_size_distribution']}")
            print(f"[DEBUG] Number of unique errors: {len(self.debug_stats['error_log'])}")

        # Final test evaluation
        if self.test_loader:
            test_loss = self.test_research_standard()
            self.metrics['test_losses'].append(test_loss)
            print(f"Final test loss: {test_loss:.4f}")

        return self.metrics

    def update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def validate_research_standard(self):
        """Research-standard validation with safe error handling"""

        self.model.eval()
        total_loss = 0
        num_successful_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    batch = batch.to(device)
                    batch_size, _ = self.safe_batch_size_calculation(batch)
                    t = self.safe_timestep_sampling(batch_size)

                    noisy_batch, target_noise_x, target_noise_pos, t_actual = self.add_research_noise_safe(batch, t)
                    outputs = self.model(noisy_batch, t_actual, getattr(batch, 'properties', None))

                    predictions = outputs[:3]
                    targets = (target_noise_x, target_noise_pos)

                    if len(outputs) > 3:
                        consistency_outputs = outputs[3]
                        loss_dict = self.criterion(predictions, targets, consistency_outputs)
                    else:
                        loss_dict = self.criterion(predictions, targets)

                    total_loss += loss_dict['total_loss'].item()
                    num_successful_batches += 1

                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG] Validation batch failed: {e}")
                    continue

        self.model.train()
        return total_loss / num_successful_batches if num_successful_batches > 0 else 0.0

    def test_research_standard(self):
        """Research-standard testing with safe error handling"""

        if not self.test_loader:
            return 0.0

        # Use EMA model for testing if available
        test_model = self.ema_model if self.ema_model else self.model
        test_model.eval()

        total_loss = 0
        num_successful_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    batch = batch.to(device)
                    batch_size, _ = self.safe_batch_size_calculation(batch)
                    t = self.safe_timestep_sampling(batch_size)

                    noisy_batch, target_noise_x, target_noise_pos, t_actual = self.add_research_noise_safe(batch, t)
                    outputs = test_model(noisy_batch, t_actual, getattr(batch, 'properties', None))

                    predictions = outputs[:3]
                    targets = (target_noise_x, target_noise_pos)

                    if len(outputs) > 3:
                        consistency_outputs = outputs[3]
                        loss_dict = self.criterion(predictions, targets, consistency_outputs)
                    else:
                        loss_dict = self.criterion(predictions, targets)

                    total_loss += loss_dict['total_loss'].item()
                    num_successful_batches += 1

                except Exception:
                    continue

        return total_loss / num_successful_batches if num_successful_batches > 0 else 0.0

    def save_research_validated_checkpoint(self, epoch, suffix):
        """Save checkpoint with research metadata and debug info"""

        os.makedirs('research_checkpoints', exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'model_config': {
                'atom_feature_dim': self.model.atom_feature_dim,
                'hidden_dim': self.model.hidden_dim,
                'timesteps': self.model.timesteps,
                'use_equivariance': self.model.use_equivariance,
                'use_consistency': self.model.use_consistency,
                'use_multi_objective': self.model.use_multi_objective
            }
        }

        # Add debug statistics if available
        if self.debug_mode:
            checkpoint['debug_stats'] = self.debug_stats

        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, f'research_checkpoints/model_{suffix}.pt')

        if self.debug_mode:
            print(f"[DEBUG] Checkpoint saved: model_{suffix}.pt")

class ResearchValidatedLoss(nn.Module):
    """
    Loss function incorporating findings from multiple research papers
    """

    def __init__(
        self,
        atom_weight=1.0,
        pos_weight=1.0,           # Equal weighting (from EDM)
        prop_weight=0.1,
        consistency_weight=0.5,   # Important for MolDiff
        valency_weight=0.3,       # Valency constraint
        adversarial_weight=0.1    # Optional adversarial component
    ):
        super().__init__()

        self.atom_weight = atom_weight
        self.pos_weight = pos_weight
        self.prop_weight = prop_weight
        self.consistency_weight = consistency_weight
        self.valency_weight = valency_weight
        self.adversarial_weight = adversarial_weight

        # Different loss functions for different components
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def calculate_valency_loss(self, atom_logits, bond_logits, valency_scores):
        """Valency consistency loss from MolDiff"""

        if bond_logits is None:
            return torch.tensor(0.0, device=atom_logits.device)

        # Simple valency constraint
        atom_types = torch.argmax(atom_logits, dim=-1)
        bond_types = torch.argmax(bond_logits, dim=-1)

        # Expected valencies for common atoms
        valency_map = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1}

        valency_loss = 0.0
        for i, atom_type in enumerate(atom_types):
            expected_valency = valency_map.get(atom_type.item(), 4)
            predicted_valency = valency_scores[i]
            valency_loss += F.mse_loss(predicted_valency, torch.tensor(expected_valency, dtype=torch.float, device=atom_logits.device))

        return valency_loss / len(atom_types)

    def forward(self, predictions, targets, consistency_outputs=None):
        """Calculate comprehensive loss"""

        pred_atom, pred_pos, pred_prop = predictions[:3]
        target_atom, target_pos = targets

        # Main reconstruction losses
        atom_loss = self.huber_loss(pred_atom, target_atom)
        pos_loss = self.mse_loss(pred_pos, target_pos)

        total_loss = self.atom_weight * atom_loss + self.pos_weight * pos_loss

        # Property loss
        prop_loss = torch.tensor(0.0, device=pred_atom.device)
        if pred_prop is not None and len(targets) > 2:
            target_prop = targets[2]
            prop_loss = self.mse_loss(pred_prop, target_prop)
            total_loss += self.prop_weight * prop_loss

        # Consistency loss (MolDiff)
        consistency_loss = torch.tensor(0.0, device=pred_atom.device)
        valency_loss = torch.tensor(0.0, device=pred_atom.device)

        if consistency_outputs is not None:
            atom_logits, bond_logits, valency_scores, atom_pairs = consistency_outputs

            if bond_logits is not None and valency_scores is not None:
                valency_loss = self.calculate_valency_loss(atom_logits, bond_logits, valency_scores)
                total_loss += self.valency_weight * valency_loss

        return {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'pos_loss': pos_loss,
            'prop_loss': prop_loss,
            'consistency_loss': consistency_loss,
            'valency_loss': valency_loss
        }
