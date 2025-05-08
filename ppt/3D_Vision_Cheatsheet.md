# 3D Vision Computing Cheatsheet

## 1. Surface Computing

### Surface Parameterization & Differential Maps
- **Ellipsoid**: $f(u,v) = (a\cos u \sin v, b\sin u \sin v, c\cos v)$
- **Differential Map**: $Df_p = \begin{bmatrix} \frac{\partial f}{\partial u} & \frac{\partial f}{\partial v} \end{bmatrix}$ = Jacobian matrix
- **Tangent Vectors**: $\vec{t}_u = \frac{\partial f}{\partial u}$, $\vec{t}_v = \frac{\partial f}{\partial v}$
- **Normal Vector**: $\vec{n} = \vec{t}_u \times \vec{t}_v$ (normalized)

### Curvature
- **Arc Length**: $s(t) = \int_0^t ||g_v'(\tau)|| d\tau$
- **Principal Normal**: $N(s) = \frac{T'(s)}{||T'(s)||}$, where $T(s) = h_v'(s)$
- **Shape Operator**: $S_p = -dN_p$, eigenvalues = principal curvatures
- **First Fundamental Form**: $I = \begin{bmatrix} E & F \\ F & G \end{bmatrix}$ where $E=\vec{t}_u \cdot \vec{t}_u$, $F=\vec{t}_u \cdot \vec{t}_v$, $G=\vec{t}_v \cdot \vec{t}_v$
- **Second Fundamental Form**: $II = \begin{bmatrix} L & M \\ M & N \end{bmatrix}$ where $L=\vec{n} \cdot \vec{t}_{uu}$, $M=\vec{n} \cdot \vec{t}_{uv}$, $N=\vec{n} \cdot \vec{t}_{vv}$
- **Gaussian Curvature**: $K = \frac{\det(II)}{\det(I)} = k_1 k_2$ (product of principal curvatures)

### Rusinkiewicz's Method for Face Curvature
1. Load vertices $p_i$, vertex normals $\vec{n}_i$, face normal $\vec{n}$
2. Select orthogonal vectors $\xi_u$, $\xi_v$ on tangent plane
3. Build $Df = [\xi_u, \xi_v]$, solve: $SDf^T(p_j - p_i) = Df^T(\vec{n}_j - \vec{n}_i)$ for $S$
4. Compute eigenvalues of $S$ for principal curvatures

## 2. 3D Geometry Processing

### Point Cloud Operations
- **Uniform Sampling**: Use Open3d/Trimesh to sample uniformly on mesh
- **Farthest Point Sampling (FPS)**:
  ```
  function FPS(P, k):
      S = [random point from P]
      for i = 1 to k-1:
          distances = min_dist(P, S)  # Min distance from each point to set S
          next_point = argmax(distances)
          S.append(P[next_point])
      return S
  ```
- **Normal Estimation by PCA**:
  1. Find k-nearest neighbors for each point
  2. Compute covariance matrix of neighbors
  3. Perform PCA (eigenvector with smallest eigenvalue = normal)
  4. Orient normals consistently (e.g., toward +Y)

## 3. Rotation

### Quaternions
- **Unit quaternion**: $q = w + xi + yj + zk$, where $w^2+x^2+y^2+z^2=1$
- **Rotation matrix from quaternion**:
  $M(q) = \begin{bmatrix} 
  1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
  2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
  2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
  \end{bmatrix}$
- **Exponential coordinates**: $\omega = \theta \mathbf{v}$ (axis-angle)
- **Quaternion from axis-angle**: $q = \cos(\theta/2) + \mathbf{v}\sin(\theta/2)$

### Skew-Symmetric Representation
- **Skew matrix**: $[\omega] = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$
- **Rodrigues' formula**: $R = \exp([\omega]) = I + \frac{\sin\theta}{\theta}[\omega] + \frac{1-\cos\theta}{\theta^2}[\omega]^2$
- **Note**: $\exp([\omega_1] + [\omega_2]) \neq \exp([\omega_1])\exp([\omega_2])$ unless $[\omega_1][\omega_2] = [\omega_2][\omega_1]$

### Double-Covering Property
- For quaternion $q$ with exponential coordinates $\omega = \theta\mathbf{v}$
- $-q$ has exponential coordinates $(2\pi-\theta)(-\mathbf{v})$

## 4. Shape Approximation

### Volume Estimation
- **Voxelization**: Count voxels × voxel volume
- **Monte Carlo with SDF**:
  1. Sample points in bounding box
  2. Classify inside/outside using SDF
  3. Volume ≈ (ratio of inside points) × (bounding box volume)

### Intersection Volume
- **Voxelization**: Count overlapping voxels × voxel volume
- **Monte Carlo**:
  1. Sample in overlapping bounding box
  2. Check points inside both meshes
  3. Volume ≈ (ratio of overlapping points) × (box volume)

### Sphere Approximation
- **Constraints**: Spheres inside mesh, fixed number N, same radius
- **Goal**: Maximize union of sphere volumes
- **Greedy Algorithm**:
  1. Place first sphere at deepest point
  2. Iteratively place next sphere at furthest point from existing spheres
- **Coverage Calculation**: Use inclusion-exclusion principle
  Volume = $\sum_i V_i - \sum_{i<j} V_{i,j} + \sum_{i<j<k} V_{i,j,k} - ...$
  where $V_{i,j,...}$ is the volume of intersection of spheres i, j, ...
- **Sphere-Sphere Intersection**:
  When distance $d < r_1+r_2$:
  $V_{overlap} = \frac{\pi(r_1+r_2-d)^2(d^2+2d(r_1+r_2)-3(r_1-r_2)^2)}{12d}$ 

## 5. Volume Rendering

### Ray Generation
- **Pixels to Rays**: Transform pixel coordinates to world space rays
- **Ray Bundle**: Contains origins, directions, sample points, sample lengths

### Point Sampling Along Rays
- **Stratified Sampler**: Uniformly sample points between near and far planes
  1. Generate uniform distances $t_i \in [t_{near}, t_{far}]$
  2. Compute points: $\mathbf{x}_i = \mathbf{o} + t_i\mathbf{d}$ (origin + distance × direction)

### Volume Integration
- **Transmittance**: $T(a,b) = \exp(-\int_a^b \sigma(t)dt)$ ≈ $\prod_{i=a}^{b-1} \exp(-\sigma_i\delta_i)$
- **Weight Calculation**: $w_i = T(t_0, t_i)(1 - \exp(-\sigma_i\delta_i))$
- **Color Integration**: $C = \sum_{i=1}^n w_i c_i$ (weighted sum of colors)
- **Depth Integration**: $D = \sum_{i=1}^n w_i t_i$ (weighted sum of distances)

### Neural Radiance Fields (NeRF)
- **Inputs**: 3D position $\mathbf{x}$ and viewing direction $\mathbf{d}$
- **Outputs**: RGB color $\mathbf{c}$ and density $\sigma$
- **Process**: 
  1. Positional encoding of inputs
  2. MLP to predict density (from position)
  3. MLP to predict color (from position and direction)
  4. Render with volume integration

## 6. Single Image to 3D

### Distance Metrics for Point Clouds
- **Chamfer Distance (CD)**:
  $d_{CD}(P, Q) = \frac{1}{|P|}\sum_{p \in P}\min_{q \in Q}||p-q||^2 + \frac{1}{|Q|}\sum_{q \in Q}\min_{p \in P}||q-p||^2$
- **Hausdorff Distance (HD)**:
  $d_{HD}(P, Q) = \max\{d(P,Q), d(Q,P)\}$
  where $d(P,Q) = \max_{p \in P}[\min_{q \in Q}||p-q||]$
- **CD vs HD**: CD is more robust to noise, HD enforces complete coverage

### Network Architectures
- **Encoder-Decoder Design**:
  1. CNN encoder: image → latent vector
  2. MLP decoder: latent vector → 3D point cloud
- **Key Components**:
  - Positional encoding
  - Skip connections
  - Multi-scale feature fusion

## 7. Surface Reconstruction

### Deep Marching Tetrahedra (DMTet)
- **Representation**: Tetrahedral grid with SDF values and deformations
- **Process**: 
  1. Initialize tetrahedral grid
  2. Optimize SDF values and vertex positions
  3. Extract mesh using marching tetrahedra

### Marching Tetrahedra Algorithm
- **Input**: Tetrahedral grid with SDF values at vertices
- **Cases**:
  1. All vertices same sign → No surface
  2. Three negative, one positive → One triangle
  3. Two positive, two negative → Two triangles
- **Edge Intersection**: $v_{ab} = \frac{v_a \cdot s(v_b) - v_b \cdot s(v_a)}{s(v_b) - s(v_a)}$

### Optimization for Surface Reconstruction
- **SDF and Deformation Parameterization**:
  1. MLP: vertex position → SDF value and deformation
  2. 3D Conv: grid → SDF values and deformations
- **Loss Functions**:
  1. Chamfer Distance loss: mesh ↔ input point cloud
  2. Laplacian regularization: improve smoothness
  
### Laplacian Regularization
- **Purpose**: Enforce mesh smoothness
- **Implementation**: Minimize vertex displacement from weighted neighborhood average
- **Formula**: $L_{reg} = \frac{1}{|V|}\sum_{v \in V}||\Delta v||^2$ where $\Delta v$ is the Laplacian at vertex v 