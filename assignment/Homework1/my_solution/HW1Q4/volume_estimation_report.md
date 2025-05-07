# Volume Estimation Method Comparison

This report compares two methods for estimating the volume of 3D meshes: Voxelization and Signed Distance Field (SDF) + Monte Carlo. The comparison focuses on accuracy (compared to `trimesh` built-in volume calculation), speed (computation time), and a qualitative discussion of memory usage.

## Methodology

- **Meshes**: Processed from the `MESH_DIR` directory as configured in the script.
- **Ground Truth**: `trimesh.Trimesh.volume` property is used as the reference volume for accuracy comparison.
- **Voxelization**: Meshes are voxelized at different target divisions along the mesh's longest dimension. Volume is calculated as `filled_voxels * pitch^3`.
- **SDF + Monte Carlo**: Points are randomly sampled within the mesh's bounding box. The Signed Distance Field (SDF) is used to classify points as inside or outside the mesh. Volume is estimated as `(points_inside / total_samples) * bounding_box_volume`.
- **Accuracy Metric**: Percentage difference from Trimesh volume: `((Estimated Volume - Trimesh Volume) / Trimesh Volume) * 100%`. Values are rounded to two decimal places.

## Results per Mesh

### Mesh: apple.obj
**Trimesh Reference Volume:** `0.000649`

#### Voxelization Results
| Target Divisions | Actual Resolution   | Pitch   | Estimated Volume | % Diff from Trimesh | Time (s) |
|------------------|---------------------|---------|------------------|---------------------|----------|
| 32               | (29, 27, 33)        | 0.00385 | 0.000190         | -70.64%             | 0.0186   |
| 64               | (57, 55, 65)        | 0.00192 | 0.000095         | -85.33%             | 0.0779   |
| 128              | (113, 107, 129)     | 0.00096 | 0.000047         | -92.71%             | 0.3289   |
| 192              | (171, 161, 193)     | 0.00064 | 0.000032         | -95.03%             | 1.0937   |

#### SDF + Monte Carlo Results (using trimesh.proximity.signed_distance)
| Sample Count | Points Inside   | BBox Volume | Estimated Volume | % Diff from Trimesh | Time (s) |
|--------------|-----------------|-------------|------------------|---------------------|----------|
| 1000         | 431             | 0.0014    | 0.000589         | -9.20%              | 1.7040                        |
| 2000         | 918             | 0.0014    | 0.000627         | -3.30%              | 3.1278                        |
| 5000         | 2416            | 0.0014    | 0.000660         | 1.80%               | 7.9325                        |
| 10000        | 4714            | 0.0014    | 0.000644         | -0.68%              | 15.1057                       |

**Brief Analysis for this Mesh (to be filled in by the user):**
- *Accuracy observations (e.g., convergence with parameters, comparison between methods for this mesh):*
- *Speed observations (e.g., how time scales with parameters, which method was faster for this mesh):*
- *Any notable issues or behaviors for this specific mesh (e.g., impact of being non-watertight if applicable):*

### Mesh: bunny.obj
**Trimesh Reference Volume:** `0.000853`

#### Voxelization Results
| Target Divisions | Actual Resolution   | Pitch   | Estimated Volume | % Diff from Trimesh | Time (s) |
|------------------|---------------------|---------|------------------|---------------------|----------|
| 32               | (33, 25, 33)        | 0.00508 | 0.000420         | -50.70%             | 0.0157   |
| 64               | (65, 51, 63)        | 0.00254 | 0.000205         | -75.98%             | 0.0856   |
| 128              | (129, 101, 125)     | 0.00127 | 0.000103         | -87.96%             | 0.3837   |
| 192              | (193, 149, 187)     | 0.00085 | 0.000069         | -91.87%             | 1.2347   |

#### SDF + Monte Carlo Results (using trimesh.proximity.signed_distance)
| Sample Count | Points Inside   | BBox Volume | Estimated Volume | % Diff from Trimesh | Time (s) |
|--------------|-----------------|-------------|------------------|---------------------|----------|
| 1000         | 283             | 0.0032    | 0.000914         | 7.18%               | 4.2633                        |
| 2000         | 539             | 0.0032    | 0.000871         | 2.07%               | 8.2801                        |
| 5000         | 1279            | 0.0032    | 0.000826         | -3.12%              | 20.0769                       |
| 10000        | 2558            | 0.0032    | 0.000826         | -3.12%              | 39.5632                       |

**Brief Analysis for this Mesh (to be filled in by the user):**
- *Accuracy observations (e.g., convergence with parameters, comparison between methods for this mesh):*
- *Speed observations (e.g., how time scales with parameters, which method was faster for this mesh):*
- *Any notable issues or behaviors for this specific mesh (e.g., impact of being non-watertight if applicable):*


## Overall Comparison and Discussion
*(This section should be completed by the user after reviewing the results across all meshes.)*

### Accuracy:
- *Discuss general trends for Voxelization accuracy as resolution changes. Does it consistently improve? What are the limits?*
- *Discuss general trends for SDF + Monte Carlo accuracy as the number of samples changes (using direct `trimesh.proximity.signed_distance`). How does it converge?*
- *Compare the overall accuracy: Which method tended to be more accurate? Were there types of meshes or scenarios where one outperformed the other?*
- *Consider the impact of mesh properties (e.g., watertightness, complexity) on accuracy for both methods.*

### Speed:
- *Discuss general trends for Voxelization speed as resolution increases.*
- *Discuss general trends for SDF + Monte Carlo speed (using direct `trimesh.proximity.signed_distance`) as the number of samples increases. Note that each call to `signed_distance` might perform its own setup, potentially impacting performance for many small batches compared to a pre-initialized `ProximityQuery`.* 
- *Compare overall speed: Which method was generally faster for comparable levels of accuracy or effort?*

### Memory Usage (Qualitative):
The script does not quantitatively measure memory usage during execution. However, qualitative observations can be made:
- **Voxelization**: Memory usage is primarily determined by the size of the voxel grid (number of voxels along each dimension). It scales with `width * height * depth`. High resolutions lead to a cubic increase in memory, which can be substantial and a limiting factor.
- **SDF + Monte Carlo (using `trimesh.proximity.signed_distance`)**:
  - *SDF Calculation*: Memory is used during the call to `trimesh.proximity.signed_distance`. While it doesn't maintain a persistent large acceleration structure like `ProximityQuery`, the internal operations for each call will consume memory based on mesh complexity and the number of query points. This approach avoids the potentially large, persistent memory footprint of a `ProximityQuery` object.
  - *Monte Carlo Sampling*: Storing the sample points and their signed distances requires memory proportional to the number of samples. This is generally less demanding than high-resolution voxel grids for typical sample counts (e.g., 10k-1M points).
For precise memory profiling, external tools or libraries (e.g., Python's `memory_profiler`, `psutil`, or system-specific monitoring tools) would need to be employed during script execution.

### Conclusions and Trade-offs:
- *Summarize the key findings from your experiments regarding accuracy, speed, and perceived memory demands.*
- *Discuss the trade-offs: For example, Voxelization might be faster at low resolutions but less accurate and very memory-hungry at high resolutions. SDF+MC might offer better accuracy for complex shapes if enough samples are used, but SDF computation can be slow/memory-intensive for highly complex meshes.*
- *Provide recommendations: When would you choose Voxelization? When would SDF + Monte Carlo be preferred? Consider factors like required accuracy, available computation time, memory constraints, and mesh characteristics.*
- *Mention any limitations of your study or challenges encountered (e.g., choice of parameters, impact of non-watertight meshes on results for both methods, range of meshes tested).*
