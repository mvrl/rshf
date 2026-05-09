"""
Simple example demonstrating EppsPulley Gaussian projection.

This shows the basic workflow of projecting a swiss roll to Gaussian.
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import lejepa


class LinearProjector(nn.Module):
    """Simple linear projection layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def main():
    print("EppsPulley Gaussian Projection - Simple Example")
    print("=" * 50)
    
    # 1. Generate swiss roll dataset
    print("\n1. Generating swiss roll dataset...")
    data, _ = make_swiss_roll(n_samples=2000, noise=0.1)
    embeddings = torch.FloatTensor(data)
    print(f"   Shape: {embeddings.shape}")
    
    # 2. Normalize
    embeddings = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-8)
    
    # 3. Create model
    print("\n2. Creating linear projection model...")
    model = LinearProjector(input_dim=3, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Create EppsPulley loss function
    print("\n3. Setting up EppsPulley loss...")
    univariate_test = lejepa.univariate.EppsPulley(n_points=17)
    loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=512
    )
    print(f"   Integration points: 17")
    print(f"   Number of slices: 512")
    
    # 5. Train
    print("\n4. Training...")
    num_epochs = 300
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        projected = model(embeddings)
        loss = loss_fn(projected)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # 6. Results
    print("\n5. Results:")
    print(f"   Initial loss: {losses[0]:.6f}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Improvement: {((losses[0] - losses[-1]) / losses[0]) * 100:.2f}%")
    
    # 7. Visualize
    print("\n6. Creating visualization...")
    with torch.no_grad():
        projected = model(embeddings)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original (3D)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    data_np = embeddings.numpy()
    ax1.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], 
                alpha=0.4, s=5, c=data_np[:, 2], cmap='viridis')
    ax1.set_title('Original: Swiss Roll (3D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Projected (2D)
    ax2 = fig.add_subplot(1, 3, 2)
    proj_np = projected.numpy()
    ax2.scatter(proj_np[:, 0], proj_np[:, 1], alpha=0.4, s=10, color='green')
    
    # Add Gaussian contours
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, 100)
    for std in [1, 2, 3]:
        ax2.plot(std * np.cos(theta), std * np.sin(theta), 'r--', 
                alpha=0.3, linewidth=1.5, label=f'{std}σ' if std == 1 else '')
    
    ax2.set_title('Projected to Gaussian (2D)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Loss curve
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(losses, linewidth=2, color='steelblue')
    ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('EppsPulley Loss')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_output.png', dpi=150, bbox_inches='tight')
    print("   Saved to: example_output.png")
    
    print("\n" + "=" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
