# STEP 1: Install packages (run this first)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torch-geometric
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
!pip install rdkit-pypi chembl-webresource-client
!pip install pandas numpy matplotlib seaborn tqdm scikit-learn

# STEP 2: Setup environment

print("\n🧪 Testing imports...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing, global_add_pool
    from torch_geometric.data import Data, DataLoader
    print(f"✅ PyTorch Geometric {torch_geometric.__version__}")
except ImportError as e:
    print(f"❌ PyTorch Geometric import failed: {e}")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    print("✅ RDKit")
except ImportError as e:
    print(f"❌ RDKit import failed: {e}")

try:
    from chembl_webresource_client.new_client import new_client
    print("✅ ChEMBL Web Resource Client")
except ImportError as e:
    print(f"❌ ChEMBL client import failed: {e}")

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    print("✅ Data science packages")
except ImportError as e:
    print(f"❌ Data science packages import failed: {e}")

# ============================================================================
# STEP 3: Setup Colab Environment
# ============================================================================

print("\n⚙️ Setting up Colab environment...")

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create necessary directories
import os
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("✅ Environment setup complete!")

# ============================================================================
# STEP 4: Test ChEMBL API Connection
# ============================================================================

print("\n🌐 Testing ChEMBL API connection...")

try:
    # Test ChEMBL API
    activity = new_client.activity

    # Try a simple query
    test_activities = activity.filter(
        target_chembl_id='CHEMBL279',  # EGFR
        standard_type='IC50'
    ).only(['molecule_chembl_id', 'canonical_smiles'])[:5]

    test_df = pd.DataFrame(test_activities)

    if len(test_df) > 0:
        print("✅ ChEMBL API connection successful!")
        print(f"   Test query returned {len(test_df)} results")
        print(f"   Sample SMILES: {test_df['canonical_smiles'].iloc[0][:50]}...")
    else:
        print("⚠️ ChEMBL API connected but no data returned")

except Exception as e:
    print(f"❌ ChEMBL API connection failed: {e}")
    print("   Will use synthetic data for training")

# ============================================================================
# STEP 5: Memory and Performance Optimization for Colab
# ============================================================================

print("\n🚀 Optimizing for Colab performance...")

# Clear any existing variables
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Set memory-efficient settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Colab-specific matplotlib settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("✅ Optimization complete!")

# ============================================================================
# STEP 6: Colab-Optimized Configuration
# ============================================================================

# Recommended settings for Colab
COLAB_CONFIG = {
    'batch_size': 8,          # Smaller batch size for Colab
    'hidden_dim': 64,         # Reduced for memory efficiency
    'num_layers': 3,          # Fewer layers for Colab
    'timesteps': 500,         # Reduced timesteps
    'num_epochs': 10,         # Fewer epochs for Colab
    'learning_rate': 1e-3,    # Slightly higher LR for faster convergence
    'max_molecules': 5000,    # Reduced dataset size for Colab
    'num_workers': 0,         # Must be 0 in Colab
    'gradient_clip': 1.0,     # Gradient clipping
    'save_interval': 2        # Save checkpoints more frequently
}

print(f"\n⚙️ Colab Configuration:")
for key, value in COLAB_CONFIG.items():
    print(f"   {key}: {value}")

# ============================================================================
# STEP 7: Colab Helper Functions
# ============================================================================

def setup_colab_training():
    """Setup training specifically optimized for Colab"""

    print("🔧 Setting up Colab-optimized training...")

    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reserve memory for training
        torch.cuda.set_per_process_memory_fraction(0.8)

    # Set environment variables for better performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    return COLAB_CONFIG

def monitor_colab_resources():
    """Monitor Colab resources during training"""

    import psutil

    # CPU usage
    cpu_percent = psutil.cpu_percent()

    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_gb = memory.used / (1024**3)

    # GPU usage (if available)
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_cached = torch.cuda.memory_reserved() / (1024**3)
        gpu_info = f"GPU Memory: {gpu_memory:.1f}GB / {gpu_cached:.1f}GB cached"

    print(f"📊 Resources: CPU {cpu_percent:.1f}% | RAM {memory_percent:.1f}% ({memory_gb:.1f}GB) | {gpu_info}")

def save_colab_checkpoint(model, epoch, train_loss, val_loss=None):
    """Save checkpoint with Colab-friendly naming"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': COLAB_CONFIG,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    checkpoint_path = f'/content/checkpoints/colab_model_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Also save to Drive if mounted
    try:
        drive_path = f'/content/drive/MyDrive/chembl_checkpoints/colab_model_epoch_{epoch}.pt'
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        torch.save(checkpoint, drive_path)
        print(f"💾 Checkpoint saved to Drive: {drive_path}")
    except:
        print(f"💾 Checkpoint saved locally: {checkpoint_path}")

# ============================================================================
# STEP 8: Mount Google Drive (Optional but Recommended)
# ============================================================================

def mount_google_drive():
    """Mount Google Drive for persistent storage"""

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")

        # Create directories in Drive
        os.makedirs('/content/drive/MyDrive/chembl_checkpoints', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/chembl_results', exist_ok=True)

        return True
    except Exception as e:
        print(f"⚠️ Google Drive mount failed: {e}")
        return False

# ============================================================================
# STEP 9: Colab-Specific Visualization
# ============================================================================

def create_colab_visualizations(train_losses, val_losses=None, generated_molecules=None):
    """Create Colab-friendly visualizations"""

    # Use Colab's inline plotting
    from IPython.display import display, HTML
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ChEMBL Diffusion Model - Colab Results', fontsize=16)

    # Training loss
    if train_losses:
        axes[0, 0].plot(train_losses, 'b-o', linewidth=2, markersize=4)
        if val_losses:
            axes[0, 0].plot(val_losses, 'r-s', linewidth=2, markersize=4)
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['Training', 'Validation'] if val_losses else ['Training'])

    # Loss improvement
    if len(train_losses) > 1:
        improvements = [train_losses[0] - loss for loss in train_losses]
        axes[0, 1].plot(improvements, 'g-', linewidth=2)
        axes[0, 1].set_title('Loss Improvement from Start')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Reduction')
        axes[0, 1].grid(True, alpha=0.3)

    # Generated molecule stats (if available)
    if generated_molecules:
        num_atoms = [mol['num_atoms'] for mol in generated_molecules]
        axes[1, 0].hist(num_atoms, bins=15, alpha=0.7, color='purple')
        axes[1, 0].set_title('Generated Molecule Sizes')
        axes[1, 0].set_xlabel('Number of Atoms')
        axes[1, 0].set_ylabel('Count')

    # Training summary
    summary_text = f"""Training on Google Colab:

• Device: {device}
• Dataset Size: {COLAB_CONFIG['max_molecules']:,} molecules
• Batch Size: {COLAB_CONFIG['batch_size']}
• Hidden Dimension: {COLAB_CONFIG['hidden_dim']}
• Training Epochs: {len(train_losses) if train_losses else 0}

Final Results:
• Final Loss: {train_losses[-1]:.4f if train_losses else 'N/A'}
• Generated Molecules: {len(generated_molecules) if generated_molecules else 0}

Status: {'✅ Training Complete' if train_losses else '⏳ Not Started'}
    """

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title('Colab Training Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('/content/colab_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 10: Complete Colab Setup Script
# ============================================================================

def complete_colab_setup():
    """Run complete setup for Colab environment"""

    print("🚀 COMPLETE GOOGLE COLAB SETUP")
    print("=" * 50)

    # Check Python version
    import sys
    print(f"Python version: {sys.version}")

    # Test all imports
    try:
        import torch
        import torch_geometric
        from rdkit import Chem
        from chembl_webresource_client.new_client import new_client
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ All packages imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device configured: {device}")

    # Mount Drive (optional)
    drive_mounted = mount_google_drive()

    # Setup directories
    os.makedirs('/content/checkpoints', exist_ok=True)
    os.makedirs('/content/results', exist_ok=True)
    os.makedirs('/content/data', exist_ok=True)
    print("✅ Directories created")

    # Test ChEMBL connection
    try:
        activity = new_client.activity
        test_query = activity.filter(target_chembl_id='CHEMBL279')[:1]
        print("✅ ChEMBL API connection successful")
    except Exception as e:
        print(f"⚠️ ChEMBL API issue: {e}")
        print("   Will use synthetic data")

    # Test basic tensor operations
    try:
        test_tensor = torch.randn(10, 5).to(device)
        test_result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ GPU tensor operations working")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

    print("\n🎉 COLAB SETUP COMPLETE!")
    print("Ready to run ChEMBL diffusion model training!")

    return True

# ============================================================================
# STEP 11: Quick Start Commands for Colab
# ============================================================================

def colab_quick_start():
    """Quick start script for immediate training"""

    print("🏃‍♂️ COLAB QUICK START")
    print("Copy and run these commands in sequence:")

    commands = [
        "# 1. Setup (run this cell first)",
        "complete_colab_setup()",
        "",
        "# 2. Load the main diffusion model code",
        "# (paste the main ChEMBL diffusion code here)",
        "",
        "# 3. Run quick test training",
        "model, losses, generated = quick_test_training()",
        "",
        "# 4. Create visualizations",
        "create_colab_visualizations(losses, generated_molecules=generated)",
        "",
        "# 5. Generate new molecules",
        "generator = MoleculeGenerator(model)",
        "new_molecules = generator.generate_molecules(num_molecules=20)",
        "evaluation = generator.evaluate_generated_molecules(new_molecules)",
        "",
        "# 6. Display results",
        "print(f'Generated {len(new_molecules)} molecules')",
        "print(f'Average validity: {np.mean(evaluation[\"validity_scores\"]):.3f}')"
    ]

    for cmd in commands:
        print(cmd)

# ============================================================================
# STEP 12: Memory Management for Colab
# ============================================================================

def optimize_colab_memory():
    """Optimize memory usage for Colab"""

    import gc

    # Clear unnecessary variables
    gc.collect()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Print memory stats
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    # Set memory-efficient training settings
    torch.backends.cudnn.benchmark = True

    print("✅ Memory optimized for Colab")

def check_colab_runtime():
    """Check Colab runtime specifications"""

    import psutil

    print("💻 COLAB RUNTIME SPECIFICATIONS")
    print("=" * 40)

    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Frequency: {cpu_freq.current:.1f} MHz" if cpu_freq else "CPU Frequency: Unknown")

    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")

    # GPU info
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
        print(f"GPU Compute: {gpu_props.major}.{gpu_props.minor}")
    else:
        print("GPU: Not available")

    # Disk space
    disk = psutil.disk_usage('/content')
    print(f"Disk Space: {disk.free / (1024**3):.1f} GB free")

# ============================================================================
# STEP 13: Execution Instructions
# ============================================================================

print("\n" + "="*60)
print("📋 GOOGLE COLAB EXECUTION INSTRUCTIONS")
print("="*60)

print("""
🔥 QUICK START (5 minutes):

1. Run this entire cell to install packages and setup
2. Copy the main ChEMBL diffusion model code to a new cell
3. Run: quick_test_training()
4. Run: create_colab_visualizations(losses, generated_molecules=generated)

📈 FULL TRAINING (30-60 minutes):

1. Complete quick start first
2. Change mode = 2 in the main code
3. Run: run_complete_chembl_training()
4. Monitor with: monitor_colab_resources()

💾 SAVING RESULTS:

• Models auto-save to /content/checkpoints/
• Mount Drive for persistent storage: mount_google_drive()
• Download results: files.download('/content/checkpoints/results.pt')

⚡ COLAB TIPS:

• Use GPU runtime: Runtime > Change runtime type > GPU
• Monitor resources: monitor_colab_resources()
• Clear memory: optimize_colab_memory()
• Save frequently to avoid session timeouts
• Use smaller batch sizes if out of memory

🔧 TROUBLESHOOTING:

• Out of memory: Reduce batch_size to 4 or 2
• Slow training: Reduce hidden_dim to 32
• Connection issues: Restart runtime and re-run setup
• ChEMBL API errors: Uses synthetic data automatically

""")

print("✅ SETUP COMPLETE - Ready to train!")
print("Run complete_colab_setup() to verify everything is working!")

# Auto-run setup
if __name__ == "__main__":
    complete_colab_setup()
    check_colab_runtime()
    colab_quick_start()
