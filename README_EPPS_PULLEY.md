# EppsPulley Gaussian Projection

This project demonstrates using the **EppsPulley univariate test** from the `lejepa` library to project various sklearn manifold datasets to Gaussian distributions using trainable linear layers.

## Overview

The EppsPulley test is a statistical test that measures how well a distribution matches an isotropic Gaussian. By using it as a loss function, we can train neural networks to transform complex, non-Gaussian distributions into Gaussian ones.

## Implementation

### Core Concept

```python
import lejepa

# Create the EppsPulley univariate test
univariate_test = lejepa.univariate.EppsPulley(n_points=17)

# Create the multivariate slicing test
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=512
)

# Train a linear projection
projected = model(embeddings)
loss = loss_fn(projected)
loss.backward()
```

### Architecture

- **Model**: Simple linear layer (`nn.Linear`)
- **Loss Function**: EppsPulley test via slicing
- **Optimizer**: Adam with learning rate 0.01
- **Training**: 300 epochs

## Datasets Tested

We tested projection on 8 different sklearn datasets:

### 3D Manifolds
1. **Swiss Roll (3D)** - Classic 3D manifold
2. **S-Curve (3D)** - 3D S-shaped manifold
3. **Blobs (3D)** - 5 clustered groups in 3D

### 2D Datasets
4. **Swiss Roll (2D)** - 2D projection of swiss roll
5. **S-Curve (2D)** - 2D projection of S-curve
6. **Circles** - Concentric circles
7. **Moons** - Two interleaving half circles
8. **Classification (4D)** - Multi-class classification dataset

## Results Summary

| Dataset | Input Dim | Initial Loss | Final Loss | Improvement |
|---------|-----------|--------------|------------|-------------|
| swiss_roll | 3 | 336.93 | 5.55 | 98.35% |
| swiss_roll_2d | 2 | 748.10 | 11.78 | 98.43% |
| s_curve | 3 | 328.56 | 2.04 | **99.38%** |
| s_curve_2d | 2 | 512.22 | 12.69 | 97.52% |
| circles | 2 | 546.93 | 5.35 | 99.02% |
| moons | 2 | 192.35 | 10.36 | 94.61% |
| blobs | 3 | 664.22 | 22.56 | 96.60% |
| classification | 4 | 556.69 | 5.30 | 99.05% |

**Average Improvement: 97.87%**

All datasets showed excellent convergence, with loss reductions of 94-99%. The S-Curve dataset achieved the best result with 99.38% improvement.

## Key Findings

1. **Linear projections are sufficient**: Even complex manifolds like swiss rolls and S-curves can be effectively projected to Gaussians using simple linear transformations.

2. **Dimensionality matters**: 2D projections of 3D manifolds (swiss_roll_2d, s_curve_2d) showed slightly different final losses compared to their 3D counterparts.

3. **Robust convergence**: All datasets converged reliably within 300 epochs, typically stabilizing around epoch 150-200.

4. **EppsPulley is effective**: The test provides a strong training signal for Gaussianization, consistently reducing loss by >94%.

## Visualizations

Each experiment generates comprehensive visualizations showing:

- **Original distribution** (3D view for 3D datasets)
- **Projected distribution** (with target Gaussian contours)
- **Training loss curve** (log scale)
- **Marginal distributions** (comparing projected vs. target Gaussian)

### Example: Swiss Roll

The swiss roll dataset (a classic 3D manifold) is successfully "unrolled" and projected to an approximately Gaussian distribution:

- Initial Loss: 336.93
- Final Loss: 5.55
- Improvement: 98.35%

See `sklearn_swiss_roll.png` for visualization.

## Files

### Main Script
- `epps_pulley_sklearn_manifolds.py` - Main implementation with sklearn datasets

### Results
- `sklearn_*.png` - Individual dataset visualizations
- `sklearn_summary.png` - Summary comparison of all experiments
- `sklearn_manifolds.log` - Complete training log

### Alternative Implementations
- `epps_pulley_gaussian_projection.py` - Original version with simple distributions
- `epps_pulley_fast.py` - Fast version with reduced epochs

## Usage

### Requirements

```bash
pip install torch lejepa scikit-learn matplotlib seaborn scipy
```

### Running Experiments

```python
python3 epps_pulley_sklearn_manifolds.py
```

This will:
1. Generate all 8 datasets
2. Train linear projections for each
3. Save visualizations to `sklearn_*.png`
4. Print summary statistics

### Custom Dataset

```python
from sklearn.datasets import make_swiss_roll

# Generate data
embeddings, _ = make_swiss_roll(n_samples=2000, noise=0.1)
embeddings = torch.FloatTensor(embeddings)

# Train projection
model, losses, projected = train_projection(
    embeddings,
    output_dim=2,
    num_epochs=300,
    lr=0.01,
    n_points=17,
    num_slices=512
)
```

## Technical Details

### EppsPulley Test

The Epps-Pulley test statistic is computed as:

```
T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt
```

where:
- `φ_empirical` is the empirical characteristic function
- `φ_normal` is the standard normal characteristic function
- `w(t)` is an integration weight
- Integration uses 17 points by default

### Slicing Approach

For multivariate data, the test:
1. Projects data onto 512 random 1D directions
2. Applies the univariate EppsPulley test to each projection
3. Aggregates results into a scalar loss

This is computationally efficient and provides good coverage of the high-dimensional space.

## References

- LeJEPA Library: https://github.com/galilai-group/lejepa
- Epps-Pulley Test: Two-sample test for univariate distributions
- Slicing Univariate Test: Extension to multivariate data via random projections

## License

Same as the parent repository.

## Contact

For questions about this implementation, please refer to the Linear issue RES-59.
