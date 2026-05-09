"""
EppsPulley Gaussian Projection with sklearn Manifold Datasets

This script uses EppsPulley to project sklearn manifold datasets 
(swiss_roll, s_curve, circles, moons, etc.) to Gaussian distributions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (
    make_swiss_roll,
    make_s_curve,
    make_circles,
    make_moons,
    make_blobs,
    make_classification
)
from scipy import stats
import lejepa

sns.set_style("whitegrid")


class LinearProjector(nn.Module):
    """Simple linear projection layer."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def generate_sklearn_dataset(dataset_name, num_samples=2000):
    """
    Generate samples from sklearn datasets.
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples to generate
    
    Returns:
        torch.Tensor: Generated samples
    """
    np.random.seed(42)
    
    if dataset_name == 'swiss_roll':
        data, _ = make_swiss_roll(n_samples=num_samples, noise=0.1)
        # Use all 3 dimensions
        
    elif dataset_name == 'swiss_roll_2d':
        data, _ = make_swiss_roll(n_samples=num_samples, noise=0.1)
        # Project to 2D (remove middle dimension)
        data = data[:, [0, 2]]
        
    elif dataset_name == 's_curve':
        data, _ = make_s_curve(n_samples=num_samples, noise=0.1)
        
    elif dataset_name == 's_curve_2d':
        data, _ = make_s_curve(n_samples=num_samples, noise=0.1)
        data = data[:, [0, 2]]
        
    elif dataset_name == 'circles':
        data, _ = make_circles(n_samples=num_samples, noise=0.05, factor=0.5)
        
    elif dataset_name == 'moons':
        data, _ = make_moons(n_samples=num_samples, noise=0.1)
        
    elif dataset_name == 'blobs':
        data, _ = make_blobs(
            n_samples=num_samples, 
            n_features=3, 
            centers=5, 
            cluster_std=1.0, 
            random_state=42
        )
        
    elif dataset_name == 'classification':
        data, _ = make_classification(
            n_samples=num_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=2,
            random_state=42
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return torch.FloatTensor(data)


def train_projection(embeddings, output_dim=2, num_epochs=300, lr=0.01, 
                     n_points=17, num_slices=512):
    """
    Train a linear projector to map embeddings to a Gaussian distribution.
    
    Args:
        embeddings: Input embeddings tensor [num_samples, input_dim]
        output_dim: Output dimensionality
        num_epochs: Number of training epochs
        lr: Learning rate
        n_points: Number of points for EppsPulley test
        num_slices: Number of slices for multivariate test
    
    Returns:
        model: Trained projection model
        losses: List of loss values during training
        projected_embeddings: Final projected embeddings
    """
    input_dim = embeddings.shape[1]
    
    # Normalize input
    embeddings_normalized = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-8)
    
    model = LinearProjector(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    univariate_test = lejepa.univariate.EppsPulley(n_points=n_points)
    loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=num_slices
    )
    
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        projected = model(embeddings_normalized)
        loss = loss_fn(projected)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    with torch.no_grad():
        projected_embeddings = model(embeddings_normalized)
    
    return model, losses, projected_embeddings


def plot_results_3d(original, projected, losses, dataset_name, save_path):
    """Create visualization for 3D datasets."""
    fig = plt.figure(figsize=(20, 5))
    
    original_np = original.detach().numpy()
    projected_np = projected.detach().numpy()
    
    # Original distribution (3D)
    ax1 = fig.add_subplot(1, 5, 1, projection='3d')
    ax1.scatter(original_np[:, 0], original_np[:, 1], original_np[:, 2], 
                alpha=0.4, s=5, c=original_np[:, 2], cmap='viridis')
    ax1.set_title(f'Original: {dataset_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Projected distribution (3D view if output is 2D, show in 3D with z=0)
    ax2 = fig.add_subplot(1, 5, 2, projection='3d')
    if projected_np.shape[1] == 2:
        ax2.scatter(projected_np[:, 0], projected_np[:, 1], np.zeros(len(projected_np)), 
                    alpha=0.4, s=5, color='green')
        ax2.set_zlabel('Z (=0)')
    else:
        ax2.scatter(projected_np[:, 0], projected_np[:, 1], projected_np[:, 2], 
                    alpha=0.4, s=5, color='green')
        ax2.set_zlabel('Z')
    ax2.set_title('Projected to Gaussian', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # 2D projection view
    ax3 = fig.add_subplot(1, 5, 3)
    if projected_np.shape[1] >= 2:
        ax3.scatter(projected_np[:, 0], projected_np[:, 1], alpha=0.4, s=10, color='green')
        theta = np.linspace(0, 2 * np.pi, 100)
        for std in [1, 2, 3]:
            ax3.plot(std * np.cos(theta), std * np.sin(theta), 'r--', alpha=0.3, linewidth=1)
    ax3.set_title('Projected (2D view)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Dimension 1')
    ax3.set_ylabel('Dimension 2')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # Loss curve
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.plot(losses, linewidth=2, color='steelblue')
    ax4.set_title('Training Loss (EppsPulley)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax5 = fig.add_subplot(1, 5, 5)
    if projected_np.shape[1] >= 2:
        kde_x = stats.gaussian_kde(projected_np[:, 0])
        kde_y = stats.gaussian_kde(projected_np[:, 1])
        
        x_range = np.linspace(-4, 4, 100)
        
        ax5.plot(x_range, kde_x(x_range), label='Projected (dim 1)', linewidth=2)
        ax5.plot(x_range, kde_y(x_range), label='Projected (dim 2)', linewidth=2)
        
        gaussian_pdf = stats.norm.pdf(x_range, 0, 1)
        ax5.plot(x_range, gaussian_pdf, 'r--', label='Target Gaussian', linewidth=2, alpha=0.7)
        
        ax5.set_title('Marginal Distributions', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Value')
        ax5.set_ylabel('Density')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    plt.close()


def plot_results_2d(original, projected, losses, dataset_name, save_path):
    """Create visualization for 2D datasets."""
    fig = plt.figure(figsize=(18, 5))
    
    original_np = original.detach().numpy()
    projected_np = projected.detach().numpy()
    
    # Original distribution
    ax1 = plt.subplot(1, 4, 1)
    ax1.scatter(original_np[:, 0], original_np[:, 1], alpha=0.5, s=10)
    ax1.set_title(f'Original: {dataset_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Projected distribution
    ax2 = plt.subplot(1, 4, 2)
    ax2.scatter(projected_np[:, 0], projected_np[:, 1], alpha=0.5, s=10, color='green')
    theta = np.linspace(0, 2 * np.pi, 100)
    for std in [1, 2, 3]:
        ax2.plot(std * np.cos(theta), std * np.sin(theta), 'r--', alpha=0.3, linewidth=1)
    ax2.set_title('Projected to Gaussian', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Loss curve
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(losses, linewidth=2, color='steelblue')
    ax3.set_title('Training Loss (EppsPulley)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax4 = plt.subplot(1, 4, 4)
    kde_x = stats.gaussian_kde(projected_np[:, 0])
    kde_y = stats.gaussian_kde(projected_np[:, 1])
    
    x_range = np.linspace(-4, 4, 100)
    
    ax4.plot(x_range, kde_x(x_range), label='Projected (dim 1)', linewidth=2)
    ax4.plot(x_range, kde_y(x_range), label='Projected (dim 2)', linewidth=2)
    
    gaussian_pdf = stats.norm.pdf(x_range, 0, 1)
    ax4.plot(x_range, gaussian_pdf, 'r--', label='Target Gaussian', linewidth=2, alpha=0.7)
    
    ax4.set_title('Marginal Distributions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    plt.close()


def run_experiments():
    """Run projection experiments on sklearn manifold datasets."""
    
    datasets = [
        ('swiss_roll', 3, True),
        ('swiss_roll_2d', 2, False),
        ('s_curve', 3, True),
        ('s_curve_2d', 2, False),
        ('circles', 2, False),
        ('moons', 2, False),
        ('blobs', 3, True),
        ('classification', 4, False),
    ]
    
    print("=" * 80)
    print("EppsPulley Gaussian Projection - sklearn Manifold Datasets")
    print("=" * 80)
    print()
    
    results_summary = []
    
    for dataset_name, input_dim, is_3d in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name.upper()} (input_dim={input_dim})")
        print(f"{'=' * 80}")
        
        embeddings = generate_sklearn_dataset(dataset_name, num_samples=2000)
        print(f"Generated {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions")
        
        # Decide output dimension
        output_dim = 2
        
        model, losses, projected = train_projection(
            embeddings,
            output_dim=output_dim,
            num_epochs=300,
            lr=0.01,
            n_points=17,
            num_slices=512
        )
        
        final_loss = losses[-1]
        initial_loss = losses[0]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"\n  Initial Loss: {initial_loss:.6f}")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        save_path = f'/workspace/sklearn_{dataset_name}.png'
        if is_3d:
            plot_results_3d(embeddings, projected, losses, dataset_name, save_path)
        else:
            plot_results_2d(embeddings, projected, losses, dataset_name, save_path)
        
        results_summary.append({
            'dataset': dataset_name,
            'input_dim': input_dim,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': improvement
        })
    
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'=' * 80}\n")
    print(f"{'Dataset':<25} {'Input Dim':<12} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['dataset']:<25} {result['input_dim']:<12} "
              f"{result['initial_loss']:<12.6f} {result['final_loss']:<12.6f} "
              f"{result['improvement']:<12.2f}%")
    
    create_summary_plot(results_summary)


def create_summary_plot(results_summary):
    """Create a summary visualization of all experiments."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [r['dataset'] for r in results_summary]
    input_dims = [r['input_dim'] for r in results_summary]
    final_losses = [r['final_loss'] for r in results_summary]
    improvements = [r['improvement'] for r in results_summary]
    
    # Final losses
    ax1 = axes[0]
    colors = ['steelblue' if dim == 2 else 'coral' if dim == 3 else 'purple' 
              for dim in input_dims]
    bars = ax1.bar(range(len(datasets)), final_losses, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.set_ylabel('Final Loss', fontsize=12)
    ax1.set_title('Final EppsPulley Loss by Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, loss) in enumerate(zip(bars, final_losses)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Improvements
    ax2 = axes[1]
    bars = ax2.bar(range(len(datasets)), improvements, color='green', alpha=0.8)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Loss Improvement by Dataset', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Input dimensions
    ax3 = axes[2]
    bars = ax3.bar(range(len(datasets)), input_dims, color='purple', alpha=0.8)
    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels(datasets, rotation=45, ha='right')
    ax3.set_ylabel('Input Dimensions', fontsize=12)
    ax3.set_title('Input Dimensionality', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, max(input_dims) + 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, dim) in enumerate(zip(bars, input_dims)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{dim}D', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/workspace/sklearn_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved summary visualization to /workspace/sklearn_summary.png")
    plt.close()


if __name__ == '__main__':
    run_experiments()
