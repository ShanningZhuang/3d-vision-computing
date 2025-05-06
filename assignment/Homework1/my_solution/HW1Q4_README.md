# Mesh Volume Estimation

This solution implements two methods for estimating the volume of a 3D mesh:

1. **Voxelization**: Converting the mesh into voxels and counting occupied voxels
2. **Signed Distance Field (SDF) + Monte Carlo**: Sampling points randomly and checking if they're inside the mesh

## Requirements

The following Python packages are required:
```
numpy
trimesh
pyvista
matplotlib
memory_profiler
psutil
```

You can install them using pip:
```
pip install numpy trimesh pyvista matplotlib memory_profiler psutil
```

## Running the Code

To run the volume estimation benchmark:

```
python HW1Q4_volume_estimation.py
```

## Output

The script will:
1. Load each mesh from the `objs_approx` directory
2. Estimate volumes using both methods with various parameters
3. Compare accuracy, speed, and memory usage
4. Generate visualization of both methods
5. Create comparison plots for both methods

Output files:
- `HW1Q4_[mesh_name]_volume_estimation_comparison.png`: Plots comparing the two methods
- `HW1Q4_[mesh_name]_voxel_[resolution].png`: Visualization of voxelization
- `HW1Q4_[mesh_name]_monte_carlo_[samples].png`: Visualization of Monte Carlo sampling

## Method Comparison

### Voxelization
- **Pros**: Precise for high resolutions, consistent results
- **Cons**: Memory usage increases cubically with resolution, slow for high resolutions

### SDF + Monte Carlo
- **Pros**: Memory efficient, can be more accurate with enough samples
- **Cons**: Probabilistic method with variance, requires good SDF computation

The comparison metrics include:
- **Accuracy**: Relative error compared to a high-resolution reference
- **Speed**: Execution time
- **Memory Usage**: Peak memory consumption

## Implementation Details

- Voxelization is implemented using trimesh's voxelization function
- SDF computation uses PyVista's implicit distance function
- Both methods use the mesh's bounding box to establish volume bounds 