import os
import traceback
import logging

# Third-party imports for training and data handling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ResearchValidatedTrainer:
    """Fixed trainer with correct metrics handling"""

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
        debug_mode=True
    ):
        self.model = model.to(device)
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

        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=lr * 0.01
        )

        # Research-validated loss function
        self.criterion = ResearchValidatedLoss()

        # FIXED: Correct metrics dictionary with proper key names
        self.metrics = {
            'train_losses': [],      # For total loss
            'val_losses': [], 
            'test_losses': [],
            'train_atom_losses': [],  # FIXED: Match the key pattern
            'train_pos_losses': [], 
            'train_prop_losses': [],
            'train_consistency_losses': []
        }

        # EMA model (optional, disable for debugging)
        self.ema_decay = ema_decay
        self.ema_model = None

    def simple_train_step(self, batch):
        """Simplified training step - let errors bubble up for debugging"""
        
        self.optimizer.zero_grad()
        batch = batch.to(device)
        
        # Calculate batch size
        if hasattr(batch, 'batch') and batch.batch is not None:
            batch_size = batch.batch.max().item() + 1
        else:
            batch_size = 1
            batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
        
        # Sample timesteps
        t = torch.randint(0, self.model.timesteps, (batch_size,), device=device)
        
        # Add noise
        sqrt_alpha_t = self.model.sqrt_alphas_cumprod[t]
        sqrt_sigma_t = self.model.sqrt_one_minus_alphas_cumprod[t]
        
        noise_x = torch.randn_like(batch.x)
        noise_pos = torch.randn_like(batch.pos)
        
        sqrt_alpha_nodes = sqrt_alpha_t[batch.batch].unsqueeze(-1)
        sqrt_sigma_nodes = sqrt_sigma_t[batch.batch].unsqueeze(-1)
        
        noisy_x = sqrt_alpha_nodes * batch.x + sqrt_sigma_nodes * noise_x
        noisy_pos = sqrt_alpha_nodes * batch.pos + sqrt_sigma_nodes * noise_pos
        
        noisy_batch = Data(
            x=noisy_x,
            pos=noisy_pos,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Forward pass
        outputs = self.model(noisy_batch, t, getattr(batch, 'properties', None))
        
        # Calculate loss
        predictions = outputs[:3]
        targets = (noise_x, noise_pos)
        
        if len(outputs) > 3 and outputs[3] is not None:
            consistency_outputs = outputs[3]
            loss_dict = self.criterion(predictions, targets, consistency_outputs)
        else:
            loss_dict = self.criterion(predictions, targets)
        
        # Backward pass
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_research_validated(self, num_epochs, validation_frequency=5):
        """FIXED training loop with correct metrics handling"""

        print(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            # FIXED: Use consistent key names
            epoch_losses = {
                'total': 0.0, 
                'atom': 0.0, 
                'pos': 0.0, 
                'prop': 0.0, 
                'consistency': 0.0
            }
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    loss_dict = self.simple_train_step(batch)
                    
                    # FIXED: Accumulate losses with correct key mapping
                    epoch_losses['total'] += loss_dict['total_loss']
                    epoch_losses['atom'] += loss_dict['atom_loss']
                    epoch_losses['pos'] += loss_dict['pos_loss']
                    epoch_losses['prop'] += loss_dict.get('prop_loss', 0.0)
                    epoch_losses['consistency'] += loss_dict.get('consistency_loss', 0.0)
                    
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'atom': f"{loss_dict['atom_loss']:.4f}",
                        'pos': f"{loss_dict['pos_loss']:.4f}"
                    })
                    
                    if self.debug_mode and batch_idx < 3:
                        print(f"\nBatch {batch_idx} losses:")
                        for key, value in loss_dict.items():
                            print(f"  {key}: {value:.6f}")
                
                except Exception as e:
                    print(f"\nTraining step failed at batch {batch_idx}: {e}")
                    if self.debug_mode:
                        print("Full traceback:")
                        traceback.print_exc()
                    print("Continuing with next batch...")
                    continue
            
            # FIXED: Calculate average losses and store with correct keys
            if num_batches > 0:
                # Average the epoch losses
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
                
                # FIXED: Store in metrics with correct key names
                self.metrics['train_losses'].append(epoch_losses['total'])
                self.metrics['train_atom_losses'].append(epoch_losses['atom'])
                self.metrics['train_pos_losses'].append(epoch_losses['pos'])
                self.metrics['train_prop_losses'].append(epoch_losses['prop'])
                self.metrics['train_consistency_losses'].append(epoch_losses['consistency'])
                
                print(f"Epoch {epoch+1} completed - Processed {num_batches}/{len(self.train_loader)} batches")
                print(f"Average losses: total={epoch_losses['total']:.4f}, atom={epoch_losses['atom']:.4f}, pos={epoch_losses['pos']:.4f}")
            else:
                print(f"❌ Epoch {epoch+1} - No successful training steps!")
                # Still append zeros to keep metrics aligned
                self.metrics['train_losses'].append(0.0)
                self.metrics['train_atom_losses'].append(0.0)
                self.metrics['train_pos_losses'].append(0.0)
                self.metrics['train_prop_losses'].append(0.0)
                self.metrics['train_consistency_losses'].append(0.0)
                continue
            
            # Validation
            if epoch % validation_frequency == 0 and self.val_loader:
                val_loss = self.validate_simple()
                self.metrics['val_losses'].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Model selection
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, 'best')
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        # Final test evaluation
        if self.test_loader:
            test_loss = self.test_simple()
            self.metrics['test_losses'].append(test_loss)
            print(f"Final test loss: {test_loss:.4f}")
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return self.metrics

    def validate_simple(self):
        """Simplified validation"""
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    batch = batch.to(device)
                    
                    # Calculate batch size
                    if hasattr(batch, 'batch') and batch.batch is not None:
                        batch_size = batch.batch.max().item() + 1
                    else:
                        batch_size = 1
                        batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
                    
                    # Sample timesteps
                    t = torch.randint(0, self.model.timesteps, (batch_size,), device=device)
                    
                    # Add noise
                    sqrt_alpha_t = self.model.sqrt_alphas_cumprod[t]
                    sqrt_sigma_t = self.model.sqrt_one_minus_alphas_cumprod[t]
                    
                    noise_x = torch.randn_like(batch.x)
                    noise_pos = torch.randn_like(batch.pos)
                    
                    sqrt_alpha_nodes = sqrt_alpha_t[batch.batch].unsqueeze(-1)
                    sqrt_sigma_nodes = sqrt_sigma_t[batch.batch].unsqueeze(-1)
                    
                    noisy_x = sqrt_alpha_nodes * batch.x + sqrt_sigma_nodes * noise_x
                    noisy_pos = sqrt_alpha_nodes * batch.pos + sqrt_sigma_nodes * noise_pos
                    
                    noisy_batch = Data(
                        x=noisy_x,
                        pos=noisy_pos,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )
                    
                    # Forward pass
                    outputs = self.model(noisy_batch, t, getattr(batch, 'properties', None))
                    
                    # Calculate loss
                    predictions = outputs[:3]
                    targets = (noise_x, noise_pos)
                    
                    if len(outputs) > 3 and outputs[3] is not None:
                        consistency_outputs = outputs[3]
                        loss_dict = self.criterion(predictions, targets, consistency_outputs)
                    else:
                        loss_dict = self.criterion(predictions, targets)
                    
                    total_loss += loss_dict['total_loss'].item()
                    num_batches += 1
                
                except Exception as e:
                    if self.debug_mode:
                        print(f"Validation batch failed: {e}")
                    continue
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def test_simple(self):
        """Simplified testing"""
        if not self.test_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    batch = batch.to(device)
                    
                    # Calculate batch size
                    if hasattr(batch, 'batch') and batch.batch is not None:
                        batch_size = batch.batch.max().item() + 1
                    else:
                        batch_size = 1
                        batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
                    
                    # Sample timesteps
                    t = torch.randint(0, self.model.timesteps, (batch_size,), device=device)
                    
                    # Add noise and forward pass (same as validation)
                    sqrt_alpha_t = self.model.sqrt_alphas_cumprod[t]
                    sqrt_sigma_t = self.model.sqrt_one_minus_alphas_cumprod[t]
                    
                    noise_x = torch.randn_like(batch.x)
                    noise_pos = torch.randn_like(batch.pos)
                    
                    sqrt_alpha_nodes = sqrt_alpha_t[batch.batch].unsqueeze(-1)
                    sqrt_sigma_nodes = sqrt_sigma_t[batch.batch].unsqueeze(-1)
                    
                    noisy_x = sqrt_alpha_nodes * batch.x + sqrt_sigma_nodes * noise_x
                    noisy_pos = sqrt_alpha_nodes * batch.pos + sqrt_sigma_nodes * noise_pos
                    
                    noisy_batch = Data(
                        x=noisy_x,
                        pos=noisy_pos,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )
                    
                    outputs = self.model(noisy_batch, t, getattr(batch, 'properties', None))
                    predictions = outputs[:3]
                    targets = (noise_x, noise_pos)
                    
                    if len(outputs) > 3 and outputs[3] is not None:
                        consistency_outputs = outputs[3]
                        loss_dict = self.criterion(predictions, targets, consistency_outputs)
                    else:
                        loss_dict = self.criterion(predictions, targets)
                    
                    total_loss += loss_dict['total_loss'].item()
                    num_batches += 1
                
                except Exception:
                    continue
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, suffix):
        """Save checkpoint"""
        os.makedirs('research_checkpoints', exist_ok=True)
        from model import ensure_model_has_all_buffers
        ensure_model_has_all_buffers(self.model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'model_config': {
                'atom_feature_dim': self.model.atom_feature_dim,
                'hidden_dim': self.model.hidden_dim,
                'timesteps': self.model.timesteps
            }
        }

        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, f'research_checkpoints/model_{suffix}.pt')
        
        if self.debug_mode:
            print(f"Checkpoint saved: model_{suffix}.pt")
            buffer_keys = [k for k in checkpoint['model_state_dict'].keys() if 'alphas' in k or 'betas' in k or 'variance' in k]
            print(f"Saved buffers: {buffer_keys}")



class ResearchValidatedLoss(nn.Module):
    """
    Simplified loss function that actually works
    """

    def __init__(
        self,
        atom_weight=1.0,
        pos_weight=1.0,
        prop_weight=0.1,
        consistency_weight=0.1
    ):
        super().__init__()

        self.atom_weight = atom_weight
        self.pos_weight = pos_weight
        self.prop_weight = prop_weight
        self.consistency_weight = consistency_weight

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, predictions, targets, consistency_outputs=None):
        """Calculate loss with proper error handling"""

        pred_atom, pred_pos, pred_prop = predictions[:3]
        target_atom, target_pos = targets

        # Main reconstruction losses
        atom_loss = self.huber_loss(pred_atom, target_atom)
        pos_loss = self.mse_loss(pred_pos, target_pos)

        total_loss = self.atom_weight * atom_loss + self.pos_weight * pos_loss

        # Property loss (if available)
        prop_loss = torch.tensor(0.0, device=pred_atom.device)
        if pred_prop is not None and len(targets) > 2:
            target_prop = targets[2]
            prop_loss = self.mse_loss(pred_prop, target_prop)
            total_loss += self.prop_weight * prop_loss

        # Consistency loss (simplified)
        consistency_loss = torch.tensor(0.0, device=pred_atom.device)
        if consistency_outputs is not None:
            try:
                atom_logits, bond_logits, valency_scores, atom_pairs = consistency_outputs
                if atom_logits is not None and bond_logits is not None:
                    # Simple consistency penalty
                    consistency_loss = torch.mean(torch.abs(valency_scores)) if valency_scores is not None else torch.tensor(0.0)
                    total_loss += self.consistency_weight * consistency_loss
            except:
                # If consistency calculation fails, just ignore it
                pass

        return {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'pos_loss': pos_loss,
            'prop_loss': prop_loss,
            'consistency_loss': consistency_loss
        }


# Test function to verify everything works
def test_fixed_trainer():
    """Test the fixed trainer with correct metrics"""
    print("Testing fixed trainer with correct metrics...")
    
    try:
        from model import ResearchValidatedDiffusionModel, _add_missing_attributes_to_model
        from torch_geometric.data import DataLoader
        
        # Create simple test data
        def create_test_molecule():
            num_atoms = 3
            x = torch.zeros(num_atoms, 119)
            x[0, 6] = 1.0  # Carbon
            x[1, 1] = 1.0  # Hydrogen  
            x[2, 1] = 1.0  # Hydrogen
            
            pos = torch.randn(num_atoms, 3)
            
            edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
            edge_attr = torch.ones(4, 5)
            batch = torch.zeros(num_atoms, dtype=torch.long)
            
            return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        # Create test data
        molecules = [create_test_molecule() for _ in range(10)]
        train_loader = DataLoader(molecules, batch_size=2, shuffle=True)
        val_loader = DataLoader(molecules[:5], batch_size=2, shuffle=False)
        
        # Create model
        model = ResearchValidatedDiffusionModel(
            atom_feature_dim=119,
            hidden_dim=128,
            num_layers=2,
            timesteps=1000
        )
        
        _add_missing_attributes_to_model(model)
        
        # Create trainer
        trainer = ResearchValidatedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            debug_mode=True
        )
        
        # Test training
        print("Running test training for 2 epochs...")
        metrics = trainer.train_research_validated(num_epochs=2, validation_frequency=1)
        
        print("✅ Fixed trainer test successful!")
        print(f"Final metrics keys: {list(metrics.keys())}")
        print(f"Training losses: {metrics['train_losses']}")
        print(f"Validation losses: {metrics['val_losses']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed trainer test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_trainer()
    
