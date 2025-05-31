# Mesh Intersection Volume Estimation Report

This report presents the results of mesh intersection volume estimation using two methods:
1. **Voxelization Method**: Voxelize both meshes and count overlapping voxels
2. **Monte Carlo Sampling**: Sample points in bounding box and check if inside both meshes

## Methodology

### Test Case Generation
For each mesh pair, we generated 6 test cases with varying intersection ratios:
- **No Intersection**: Meshes separated with no overlap (target ratio: 0%)
- **Small Intersection**: Small overlap (target ratio: ~15%)
- **Medium Intersection**: Medium overlap (target ratio: ~40%)
- **Large Intersection**: Large overlap (target ratio: ~70%)
- **Maximum Intersection**: One mesh inside another (target ratio: ~90%)
- **Rotated Intersection**: Meshes with rotation and medium overlap (target ratio: ~30%)

### Voxelization Method
1. Compute combined bounding box of both meshes
2. Voxelize both meshes using the same pitch and coordinate system
3. Fill voxel grids to create solid volumes
4. Compute intersection as overlapping voxels
5. Volume = (Number of overlapping voxels) × (Voxel size)³

### Monte Carlo Method
1. Compute combined bounding box of both meshes
2. Randomly sample points within the bounding box
3. Use Signed Distance Field (SDF) to check if points are inside both meshes
4. Volume ≈ (Fraction of points inside both) × (Bounding box volume)

## Results by Mesh Pair

### apple.obj & bunny.obj
**Mesh 1 Volume**: 0.000649 | **Mesh 2 Volume**: 0.000853

#### Test Case: no_intersection
**Description**: Meshes separated with no overlap
**Expected Intersection Ratio**: 0.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000000 | 0 | 0.0536 | 399.65 |
| 64 | 0.000000 | 0 | 0.0463 | 395.50 |
| 128 | 0.000000 | 0 | 0.0472 | 395.50 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000000 | 0 | 0.0282 | 2.4654 | 487.28 |
| 5000 | 0.000000 | 0 | 0.0282 | 10.1257 | 578.68 |
| 10000 | 0.000000 | 0 | 0.0282 | 20.6176 | 524.54 |

#### Test Case: small_intersection
**Description**: Small overlap between meshes
**Expected Intersection Ratio**: 15.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000205 | 615 | 0.0539 | 555.41 |
| 64 | 0.000164 | 3935 | 0.1519 | 586.16 |
| 128 | 0.000145 | 27930 | 0.7397 | 591.29 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000154 | 35 | 0.0044 | 5.2110 | 729.60 |
| 5000 | 0.000134 | 152 | 0.0044 | 24.6291 | 743.45 |
| 10000 | 0.000132 | 300 | 0.0044 | 51.1219 | 751.18 |

#### Test Case: medium_intersection
**Description**: Medium overlap between meshes
**Expected Intersection Ratio**: 40.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000402 | 2086 | 0.0541 | 751.18 |
| 64 | 0.000342 | 14227 | 0.2289 | 725.74 |
| 128 | 0.000315 | 104852 | 1.0932 | 810.86 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000279 | 76 | 0.0037 | 5.6368 | 866.14 |
| 5000 | 0.000294 | 401 | 0.0037 | 26.1929 | 827.44 |
| 10000 | 0.000304 | 829 | 0.0037 | 52.6464 | 817.57 |

#### Test Case: large_intersection
**Description**: Large overlap between meshes
**Expected Intersection Ratio**: 70.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000502 | 3825 | 0.0558 | 817.57 |
| 64 | 0.000444 | 27068 | 0.2819 | 861.38 |
| 128 | 0.000416 | 202759 | 1.3765 | 872.94 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000443 | 137 | 0.0032 | 5.6918 | 845.11 |
| 5000 | 0.000429 | 664 | 0.0032 | 26.9878 | 861.27 |
| 10000 | 0.000400 | 1238 | 0.0032 | 54.6940 | 873.01 |

#### Test Case: max_intersection
**Description**: Maximum overlap - smaller mesh inside larger
**Expected Intersection Ratio**: 90.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000164 | 1248 | 0.0540 | 873.01 |
| 64 | 0.000139 | 8498 | 0.2064 | 825.04 |
| 128 | 0.000128 | 62549 | 0.9812 | 832.22 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000123 | 38 | 0.0032 | 7.2220 | 955.30 |
| 5000 | 0.000117 | 181 | 0.0032 | 34.0067 | 1022.23 |
| 10000 | 0.000110 | 340 | 0.0032 | 67.9208 | 1034.45 |

#### Test Case: rotated_intersection
**Description**: Meshes with rotation and medium overlap
**Expected Intersection Ratio**: 30.0%

**Voxelization Results:**

| Resolution | Volume | Voxel Count | Time (s) | Memory (MB) |
|------------|--------|-------------|----------|-------------|
| 32 | 0.000519 | 2497 | 0.0550 | 1034.70 |
| 64 | 0.000443 | 17072 | 0.2174 | 972.79 |
| 128 | 0.000409 | 126041 | 1.0390 | 1051.63 |

**Monte Carlo Results:**

| Samples | Volume | Inside Count | BBox Volume | Time (s) | Memory (MB) |
|---------|--------|--------------|-------------|----------|-------------|
| 1000 | 0.000425 | 98 | 0.0043 | 6.8107 | 1041.26 |
| 5000 | 0.000398 | 459 | 0.0043 | 31.8059 | 1049.86 |
| 10000 | 0.000387 | 893 | 0.0043 | 64.8282 | 1055.29 |

## Analysis and Discussion

### Accuracy Analysis
*(To be completed based on experimental results)*

- **Voxelization Accuracy**: How does accuracy change with resolution?
- **Monte Carlo Accuracy**: How does accuracy improve with sample count?
- **Method Comparison**: Which method provides better accuracy for different intersection scenarios?
- **Edge Cases**: How do methods perform with no intersection vs. maximum intersection?

### Computational Cost Analysis
*(To be completed based on experimental results)*

- **Voxelization Performance**: How does computation time scale with resolution?
- **Monte Carlo Performance**: How does computation time scale with sample count?
- **Memory Usage**: Which method is more memory efficient?
- **Speed vs. Accuracy Trade-offs**: Optimal parameters for different use cases

### Method Comparison Summary
*(To be completed based on experimental results)*

| Aspect | Voxelization | Monte Carlo |
|--------|--------------|-------------|
| **Accuracy** | [Analysis needed] | [Analysis needed] |
| **Speed** | [Analysis needed] | [Analysis needed] |
| **Memory** | [Analysis needed] | [Analysis needed] |
| **Scalability** | [Analysis needed] | [Analysis needed] |
| **Best Use Case** | [Analysis needed] | [Analysis needed] |

### Conclusions
*(To be completed based on experimental results)*

- Key findings from the intersection volume estimation experiments
- Recommendations for method selection based on requirements
- Limitations and potential improvements
