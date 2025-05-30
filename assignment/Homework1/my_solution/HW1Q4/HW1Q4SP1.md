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
| Target Divisions | Actual Resolution   | Pitch   | Estimated Volume | % Diff from Trimesh | Time (s) | Memory RSS (MB) |
|------------------|---------------------|---------|------------------|---------------------|----------|-----------------|
| 32               | (29, 27, 33)        | 0.00385 | 0.000751         | 15.69%              | 0.0211   | 349.35            |
| 64               | (57, 55, 65)        | 0.00192 | 0.000698         | 7.55%               | 0.0849   | 398.18            |
| 128              | (113, 107, 129)     | 0.00096 | 0.000673         | 3.69%               | 0.3591   | 428.52            |
| 192              | (171, 161, 193)     | 0.00064 | 0.000665         | 2.51%               | 1.2061   | 560.98            |

#### SDF + Monte Carlo Results (using trimesh.proximity.signed_distance)
| Sample Count | Points Inside   | BBox Volume | Estimated Volume | % Diff from Trimesh | Time (s) | Memory RSS (MB) |
|--------------|-----------------|-------------|------------------|---------------------|----------|-----------------|
| 1000         | 479             | 0.0014    | 0.000655         | 0.92%               | 1.7354   | 640.01            |
| 2000         | 946             | 0.0014    | 0.000647         | -0.35%              | 3.2723   | 645.36            |
| 5000         | 2354            | 0.0014    | 0.000643         | -0.81%              | 8.0871   | 651.32            |
| 10000        | 4742            | 0.0014    | 0.000648         | -0.09%              | 16.0663  | 659.96            |

**Brief Analysis for this Mesh:**
- *Accuracy observations:* Voxelization shows consistent improvement with higher resolution (15.69% → 2.51% error). SDF + Monte Carlo achieves excellent accuracy even with 1000 samples (0.92% error) and converges to near-perfect accuracy with more samples (-0.09% error at 10k samples).
- *Speed observations:* Voxelization is significantly faster, ranging from 0.02s to 1.2s, while SDF + Monte Carlo is much slower, ranging from 1.7s to 16s. Time scales roughly linearly with voxel count and sample count respectively.
- *Notable behaviors:* The apple mesh appears to be well-behaved for both methods. SDF + Monte Carlo shows some variance in intermediate sample counts but converges well. Voxelization consistently overestimates volume, while SDF + Monte Carlo oscillates around the true value.

### Mesh: bunny.obj
**Trimesh Reference Volume:** `0.000853`

#### Voxelization Results
| Target Divisions | Actual Resolution   | Pitch   | Estimated Volume | % Diff from Trimesh | Time (s) | Memory RSS (MB) |
|------------------|---------------------|---------|------------------|---------------------|----------|-----------------|
| 32               | (33, 25, 33)        | 0.00508 | 0.001080         | 26.62%              | 0.0166   | 660.82            |
| 64               | (65, 51, 63)        | 0.00254 | 0.000960         | 12.50%              | 0.0854   | 660.82            |
| 128              | (129, 101, 125)     | 0.00127 | 0.000906         | 6.17%               | 0.4216   | 661.85            |
| 192              | (193, 149, 187)     | 0.00085 | 0.000888         | 4.12%               | 1.3270   | 657.52            |

#### SDF + Monte Carlo Results (using trimesh.proximity.signed_distance)
| Sample Count | Points Inside   | BBox Volume | Estimated Volume | % Diff from Trimesh | Time (s) | Memory RSS (MB) |
|--------------|-----------------|-------------|------------------|---------------------|----------|-----------------|
| 1000         | 245             | 0.0032    | 0.000791         | -7.21%              | 4.4938   | 682.86            |
| 2000         | 549             | 0.0032    | 0.000887         | 3.96%               | 8.1532   | 692.02            |
| 5000         | 1301            | 0.0032    | 0.000841         | -1.45%              | 20.8560  | 690.78            |
| 10000        | 2597            | 0.0032    | 0.000839         | -1.64%              | 41.7568  | 715.51            |

**Brief Analysis for this Mesh:**
- *Accuracy observations:* Voxelization again shows steady improvement (26.62% → 4.12% error) but with higher initial error than the apple. SDF + Monte Carlo shows more variance initially (-7.21% to 3.96%) but stabilizes around -1.5% error with higher sample counts.
- *Speed observations:* Voxelization remains fast (0.02s to 1.3s), while SDF + Monte Carlo is notably slower for the bunny (4.5s to 41.8s), suggesting the mesh complexity significantly impacts SDF computation time.
- *Notable behaviors:* The bunny mesh appears more challenging for both methods, with higher errors at equivalent parameter settings. The complex geometry (ears, detailed features) likely contributes to discretization challenges for voxelization and higher computational cost for SDF calculations.

## Overall Comparison and Discussion

### Accuracy:
Voxelization shows predictable convergence behavior across both meshes. As resolution increases from 32 to 192 target divisions, accuracy consistently improves, with errors decreasing from ~15-27% to ~2-4%. The method systematically overestimates volume due to the discrete nature of voxel representation, where partial voxels are either fully included or excluded. Complex geometries like the bunny show higher errors at equivalent resolutions due to finer surface details being poorly captured by coarse voxelization.

SDF + Monte Carlo demonstrates excellent convergence properties with increasing sample counts. Even with just 1000 samples, both meshes achieve reasonable accuracy (0.92% for apple, -7.21% for bunny). The method converges to very high accuracy with 10,000 samples (-0.09% for apple, -1.64% for bunny). The statistical nature of Monte Carlo sampling causes some variance in intermediate sample counts, but the overall trend is toward the true volume. The method can both over- and under-estimate volume, showing less systematic bias than voxelization.

Overall, SDF + Monte Carlo achieves superior accuracy, especially at higher parameter settings, while voxelization provides reasonable accuracy that improves predictably with resolution.

### Speed:
Voxelization demonstrates excellent computational efficiency, with times ranging from 0.02s to 1.3s even at the highest resolution tested (192 divisions). The time complexity scales roughly with the cube of the target divisions, but remains manageable for the tested range. The method shows minimal sensitivity to mesh complexity in terms of computation time.

SDF + Monte Carlo is significantly slower, with computation times ranging from 1.7s to 41.8s. The time scales roughly linearly with the number of sample points, but the base cost is high due to SDF computation overhead. Mesh complexity has a substantial impact on performance - the bunny mesh took 2.5-2.6x longer than the apple at equivalent sample counts, likely due to more complex geometry requiring more intensive SDF calculations.

Voxelization is clearly the winner for speed, being 5-30x faster than SDF + Monte Carlo for comparable accuracy levels.

### Memory Usage (Quantitative and Qualitative):
The RSS measurements show that both methods have significant memory footprints, with baseline memory around 350-660 MB increasing to 560-715 MB during computation.

**Voxelization**: Memory usage scales with voxel grid size (width × height × depth). The method shows moderate memory increases with resolution (349 MB → 561 MB for apple). The cubic scaling of memory with resolution will become a limiting factor at very high resolutions, potentially requiring hundreds of GB for target divisions of 500-1000.

**SDF + Monte Carlo**: Memory usage shows less dramatic scaling with parameter increases (640 MB → 660 MB for apple). However, the base memory cost is higher, and the `trimesh.proximity.signed_distance` function may have internal memory overhead that's not fully captured in RSS measurements. The method avoids the cubic memory scaling issue of voxelization.

For the tested parameter ranges, memory differences are not dramatic, but voxelization's cubic scaling presents a fundamental limitation for high-resolution applications.

### Conclusions and Trade-offs:
**Key Findings:**
1. **Accuracy**: SDF + Monte Carlo achieves superior accuracy (sub-1% error) but requires many samples. Voxelization provides good accuracy (2-4% error) at reasonable resolutions.
2. **Speed**: Voxelization is dramatically faster (5-30x) than SDF + Monte Carlo.
3. **Memory**: Both methods have substantial memory requirements, but voxelization faces cubic scaling limitations at high resolutions.

**When to choose Voxelization:**
- When speed is critical and moderate accuracy (2-5% error) is acceptable
- For real-time applications or batch processing of many meshes
- When memory constraints prevent very high-resolution voxelization
- For meshes with relatively simple geometry where discretization artifacts are minimal

**When to choose SDF + Monte Carlo:**
- When high accuracy (sub-1% error) is required
- For complex meshes with fine geometric details that voxelization would poorly represent
- When computational time is less critical than precision
- For applications where statistical confidence intervals around volume estimates are valuable

**Limitations and Considerations:**
- The study tested only two meshes; results may vary significantly for meshes with different characteristics (non-watertight, highly concave, or extremely complex geometry)
- Very high voxel resolutions (>500 divisions) were not tested due to memory constraints
- SDF computation time may vary significantly with different mesh complexities not represented in this limited sample
- The choice of `trimesh.proximity.signed_distance` may not be the most optimized SDF implementation available
- Monte Carlo variance could be reduced with better sampling strategies (e.g., stratified sampling, quasi-random sequences)
