import trimesh
import numpy as np
import time # Optional: to time the execution
# Add imports for KDTree and PCA
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
# Add imports for Q4
import matplotlib.pyplot as plt
import os # To create output directory if needed

# ==============================================================================
# --- Part 1: Uniform Sampling (saddle.obj) ---
# ==============================================================================
print("--- Processing Part 1-3 (saddle.obj) ---")
# Define the path to your mesh file
mesh_file_saddle = '../saddle.obj'
num_points_to_sample = 100000

# Load the mesh from the .obj file
# Ensure 'saddle.obj' is in the correct path relative to your script,
# or provide the absolute path.
try:
    mesh_saddle = trimesh.load_mesh(mesh_file_saddle)
    print(f"Successfully loaded mesh: {mesh_file_saddle}")
except ValueError as e:
    print(f"Error loading mesh '{mesh_file_saddle}': {e}")
    print("Please ensure the file exists and is a valid mesh format.")
    exit() # Exit if the mesh cannot be loaded

# Check if the mesh has faces, required for surface sampling
if not hasattr(mesh_saddle, 'faces') or not mesh_saddle.faces.shape[0] > 0:
    print(f"Error: Mesh '{mesh_file_saddle}' has no faces. Cannot sample surface points.")
    exit()

# Sample points uniformly from the surface
# The trimesh.sample.sample_surface() function returns points and face indices.
# We only need the points here.
points, _ = trimesh.sample.sample_surface(mesh_saddle, num_points_to_sample)

# 'points' is a numpy array of shape (num_points_to_sample, 3)
print(f"Successfully sampled {points.shape[0]} points from the mesh surface.")

# You can now use the 'points' array for further processing.
# For example, print the first 5 sampled points:
print("First 5 sampled points:\n", points[:5])

# --- Save the 100k points ---
output_npy_file_100k = '../saddle_sampled_100k_points.npy'
np.save(output_npy_file_100k, points)
print(f"Sampled 100k points saved to {output_npy_file_100k}")

output_ply_file_100k = '../saddle_sampled_100k_points.ply'
point_cloud_100k = trimesh.points.PointCloud(points)
try:
    point_cloud_100k.export(output_ply_file_100k)
    print(f"Sampled 100k points saved to {output_ply_file_100k}")
except Exception as e:
    print(f"Error saving 100k PLY: {e}")


# ==============================================================================
# --- Part 2: Farthest Point Sampling (saddle.obj) ---
# ==============================================================================

def farthest_point_sampling(points, num_samples):
    """
    Performs Iterative Farthest Point Sampling (FPS) on a point cloud.

    Args:
        points (np.ndarray): Input point cloud, shape (N, 3).
        num_samples (int): The number of points to sample (k).

    Returns:
        np.ndarray: The sampled points, shape (num_samples, 3).
        np.ndarray: Indices of the sampled points in the original array, shape (num_samples,).
    """
    num_points = points.shape[0]
    if num_samples <= 0 or num_samples > num_points:
        raise ValueError("num_samples must be between 1 and the total number of points.")

    # Array to store the indices of the sampled points
    sampled_indices = np.zeros(num_samples, dtype=np.int32)

    # Array to store the minimum squared distance from each point to any sampled point
    min_distances_sq = np.full(num_points, np.inf, dtype=np.float32)

    # --- Step 1: Choose the first point ---
    first_point_idx = 0
    sampled_indices[0] = first_point_idx

    # --- Step 2: Calculate initial distances ---
    last_sampled_point = points[first_point_idx:first_point_idx+1, :]
    dists_sq = np.sum((points - last_sampled_point)**2, axis=1)
    min_distances_sq = dists_sq
    min_distances_sq[first_point_idx] = -1.0 # Mark as sampled

    # --- Step 3: Iteratively sample remaining points ---
    for i in range(1, num_samples):
        farthest_point_idx = np.argmax(min_distances_sq)
        sampled_indices[i] = farthest_point_idx
        last_sampled_point = points[farthest_point_idx:farthest_point_idx+1, :]
        new_dists_sq = np.sum((points - last_sampled_point)**2, axis=1)
        min_distances_sq = np.minimum(min_distances_sq, new_dists_sq)
        min_distances_sq[farthest_point_idx] = -1.0

    sampled_points = points[sampled_indices]
    return sampled_points, sampled_indices

# --- Execute FPS ---
num_target_samples = 4000

if points.shape[0] < num_target_samples:
     print(f"Warning: Number of initial points ({points.shape[0]}) is less than target FPS samples ({num_target_samples}).")
     num_target_samples = points.shape[0]

print(f"\nStarting Farthest Point Sampling for {num_target_samples} points...")
start_time_fps = time.time()

sampled_points_fps, _ = farthest_point_sampling(points, num_target_samples)

end_time_fps = time.time()
print(f"Finished FPS in {end_time_fps - start_time_fps:.2f} seconds.")
print(f"Shape of sampled points: {sampled_points_fps.shape}")
print("First 5 FPS sampled points:\n", sampled_points_fps[:5])

# --- Save the FPS sampled points ---
output_fps_npy_file = '../saddle_fps_4k_points.npy'
np.save(output_fps_npy_file, sampled_points_fps)
print(f"FPS sampled points saved to {output_fps_npy_file}")

output_fps_ply_file = '../saddle_fps_4k_points.ply'
point_cloud_fps = trimesh.points.PointCloud(sampled_points_fps)
try:
    point_cloud_fps.export(output_fps_ply_file)
    print(f"FPS sampled points saved to {output_fps_ply_file}")
except Exception as e:
    print(f"Error saving FPS PLY: {e}")


# ==============================================================================
# --- Part 3: Normal Estimation using PCA (saddle.obj) ---
# ==============================================================================

def estimate_normals_pca(query_points, source_points, k_neighbors=50):
    """
    Estimates normals for query points using PCA on k nearest neighbors
    from source points.
    """
    print(f"\nBuilding KDTree on {source_points.shape[0]} source points...")
    start_time_kdtree = time.time()
    kdt = KDTree(source_points)
    end_time_kdtree = time.time()
    print(f"KDTree built in {end_time_kdtree - start_time_kdtree:.2f} seconds.")

    print(f"Querying {k_neighbors} nearest neighbors for {query_points.shape[0]} query points...")
    start_time_query = time.time()
    distances, indices = kdt.query(query_points, k=k_neighbors)
    end_time_query = time.time()
    print(f"Neighbor query finished in {end_time_query - start_time_query:.2f} seconds.")

    estimated_normals = np.zeros_like(query_points)
    print(f"Estimating normals using PCA (k={k_neighbors})...")
    start_time_pca = time.time()
    pca = PCA(n_components=3)

    for i in range(query_points.shape[0]):
        neighbor_points = source_points[indices[i]]
        pca.fit(neighbor_points)
        normal = pca.components_[-1]
        if np.dot(normal, [0, 1, 0]) < 0:
            normal = -normal
        estimated_normals[i] = normal

    end_time_pca = time.time()
    print(f"Normal estimation finished in {end_time_pca - start_time_pca:.2f} seconds.")
    return estimated_normals

# --- Execute Normal Estimation ---
k_neighbors_for_pca = 50
estimated_normals = estimate_normals_pca(sampled_points_fps, points, k_neighbors=k_neighbors_for_pca)

print(f"\nShape of estimated normals: {estimated_normals.shape}")
print("First 5 estimated normals:\n", estimated_normals[:5])

# --- Normalize and Save Normals ---
norms = np.linalg.norm(estimated_normals, axis=1, keepdims=True)
norms[norms == 0] = 1.0
unit_normals = estimated_normals / norms

# Save as PLY file with normals
output_ply_with_normals = '../saddle_fps_4k_points_with_normals.ply'
point_cloud_with_normals = trimesh.points.PointCloud(vertices=sampled_points_fps, vertex_normals=unit_normals)
try:
    point_cloud_with_normals.export(output_ply_with_normals)
    print(f"FPS points with estimated normals saved to {output_ply_with_normals}")
except Exception as e:
    print(f"Error saving PLY with normals: {e}")

# Save normals separately as NPY
output_normals_npy_file = '../saddle_fps_4k_normals.npy'
np.save(output_normals_npy_file, unit_normals)
print(f"Estimated normals saved to {output_normals_npy_file}")


# ==============================================================================
# --- Part 4: Mesh Curvature Estimation (Rusinkiewicz's method) ---
# ==============================================================================
print("\n--- Processing Part 4 (icosphere.obj, sievert.obj) ---")

def compute_face_curvatures_rusinkiewicz(mesh_file):
    """
    Computes principal curvatures (k1, k2) and Gaussian curvature (K)
    for each face of a mesh using Rusinkiewicz's method.
    """
    try:
        mesh = trimesh.load_mesh(mesh_file, process=True)
        print(f"\nSuccessfully loaded mesh: {mesh_file}")
        print(f"  Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")
    except Exception as e:
        print(f"Error loading mesh '{mesh_file}': {e}")
        return None, None, None

    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals.shape[0] != mesh.vertices.shape[0]:
        print(f"Error: Vertex normals not found or computed for {mesh_file}.")
        try:
            mesh.vertex_normals # Force computation if possible
            print("Computed vertex normals.")
            if mesh.vertex_normals.shape[0] != mesh.vertices.shape[0]: raise ValueError("Normal count mismatch.")
        except Exception as e:
            print(f"Failed to ensure vertex normals: {e}")
            return None, None, None

    num_faces = mesh.faces.shape[0]
    principal_k1 = np.full(num_faces, np.nan, dtype=np.float64)
    principal_k2 = np.full(num_faces, np.nan, dtype=np.float64)
    gaussian_K = np.full(num_faces, np.nan, dtype=np.float64)
    print("Computing face curvatures...")
    tiny_float = 1e-12

    for face_idx, face in enumerate(mesh.faces):
        p0, p1, p2 = mesh.vertices[face]
        n0, n1, n2 = mesh.vertex_normals[face]

        e0 = p1 - p0
        norm_e0 = np.linalg.norm(e0)
        if norm_e0 < tiny_float: continue
        xi_u = e0 / norm_e0

        face_normal = mesh.face_normals[face_idx].copy()
        norm_fn = np.linalg.norm(face_normal)
        if norm_fn < tiny_float: continue
        face_normal /= norm_fn

        xi_v = np.cross(face_normal, xi_u)
        norm_xi_v = np.linalg.norm(xi_v)
        if norm_xi_v < tiny_float: continue
        xi_v /= norm_xi_v

        Df = np.column_stack((xi_u, xi_v))

        e1 = p2 - p1
        e2 = p0 - p2
        dn0 = n1 - n0
        dn1 = n2 - n1
        dn2 = n0 - n2

        v0 = Df.T @ e0; v1 = Df.T @ e1; v2 = Df.T @ e2
        w0 = Df.T @ dn0; w1 = Df.T @ dn1; w2 = Df.T @ dn2

        M = np.zeros((6, 4), dtype=np.float64)
        b = np.zeros(6, dtype=np.float64)

        M[0, 0]=v0[0]; M[0, 1]=v0[1]; M[1, 2]=v0[0]; M[1, 3]=v0[1]; b[0]=w0[0]; b[1]=w0[1]
        M[2, 0]=v1[0]; M[2, 1]=v1[1]; M[3, 2]=v1[0]; M[3, 3]=v1[1]; b[2]=w1[0]; b[3]=w1[1]
        M[4, 0]=v2[0]; M[4, 1]=v2[1]; M[5, 2]=v2[0]; M[5, 3]=v2[1]; b[4]=w2[0]; b[5]=w2[1]

        try:
            s_vec, residuals, rank, s_singular_values = np.linalg.lstsq(M, b, rcond=None)
            if rank != 4: continue
            S = s_vec.reshape((2, 2))
            S = 0.5 * (S + S.T) # Symmetrize
            eigenvalues = np.linalg.eigvalsh(S)
            kvals = np.sort(eigenvalues)[::-1]
            principal_k1[face_idx] = kvals[0]
            principal_k2[face_idx] = kvals[1]
            gaussian_K[face_idx] = kvals[0] * kvals[1]
        except np.linalg.LinAlgError:
            continue
        except Exception:
             continue

    valid_indices = ~np.isnan(principal_k1)
    print(f"Computed curvatures for {np.sum(valid_indices)} / {num_faces} faces.")
    return principal_k1[valid_indices], principal_k2[valid_indices], gaussian_K[valid_indices]

def plot_curvature_histograms(k1, k2, K, mesh_name, output_dir='../curvature_results'):
    """Plots histograms for principal and Gaussian curvatures."""
    if k1 is None or k1.size == 0:
        print(f"No valid curvature data to plot for {mesh_name}.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir); print(f"Created output directory: {output_dir}")

    output_file = os.path.join(output_dir, f'{mesh_name}_curvature_histograms.png')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Curvature Histograms for {mesh_name}', fontsize=16)

    lim_k1 = np.percentile(k1, [1, 99]); lim_k2 = np.percentile(k2, [1, 99]); lim_K = np.percentile(K, [1, 99])
    if lim_k1[0] == lim_k1[1]: lim_k1[1] += 1e-6
    if lim_k2[0] == lim_k2[1]: lim_k2[1] += 1e-6
    if lim_K[0] == lim_K[1]: lim_K[1] += 1e-6

    axes[0].hist(k1, bins=50, range=lim_k1, color='skyblue', edgecolor='black'); axes[0].set_title('Principal Curvature k1'); axes[0].set_xlabel('Curvature Value'); axes[0].set_ylabel('Frequency'); axes[0].grid(axis='y', alpha=0.75)
    axes[1].hist(k2, bins=50, range=lim_k2, color='lightcoral', edgecolor='black'); axes[1].set_title('Principal Curvature k2'); axes[1].set_xlabel('Curvature Value'); axes[1].grid(axis='y', alpha=0.75)
    axes[2].hist(K, bins=50, range=lim_K, color='lightgreen', edgecolor='black'); axes[2].set_title('Gaussian Curvature K = k1 * k2'); axes[2].set_xlabel('Curvature Value'); axes[2].grid(axis='y', alpha=0.75)

    median_K = np.median(K)
    print(f"  Median Gaussian Curvature (K) for {mesh_name}: {median_K:.4f}")
    axes[2].axvline(median_K, color='red', linestyle='dashed', linewidth=1.5, label=f'Median K = {median_K:.3f}')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    print(f"Saved histogram plot to {output_file}")
    plt.close(fig)

# --- Execute Curvature Estimation ---
# Assuming the script is in my_solution, the data is one level up
mesh_files_curvature = ['../icosphere.obj', '../sievert.obj']
output_curvature_dir = '../curvature_results' # Define output dir here

for file_path in mesh_files_curvature:
    if not os.path.exists(file_path):
        print(f"Error: Mesh file not found at {file_path}")
        continue
    mesh_name = os.path.splitext(os.path.basename(file_path))[0]
    k1_vals, k2_vals, K_vals = compute_face_curvatures_rusinkiewicz(file_path)
    plot_curvature_histograms(k1_vals, k2_vals, K_vals, mesh_name, output_dir=output_curvature_dir)

print("\n--- All processing finished ---")
