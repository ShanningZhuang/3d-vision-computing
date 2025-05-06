import trimesh
import numpy as np
import time
import os

# --- Configuration ---
# Adjust MESH_DIR if your 'objs_approx' folder is located elsewhere relative to this script.
# Assumes this script is in a 'my_solution' type folder, and 'objs_approx' is a sibling to its parent.
# e.g., script is in '.../Homework1/my_solution/' and meshes are in '.../Homework1/objs_approx/'
MESH_DIR = "../../objs_approx"

# Resolutions for Voxelization: number of divisions along the mesh's longest dimension.
# Higher numbers mean finer resolution.
VOXEL_RESOLUTIONS_DIVISIONS = [32, 64, 128, 192] # Reduced 256 due to potential memory/time

# Number of samples for SDF + Monte Carlo.
MONTE_CARLO_SAMPLE_COUNTS = [10000, 20000, 50000] # Reduced 1M for quicker runs

# --- Helper Functions ---
def get_mesh_files(directory):
    """Gets all common mesh files from a directory."""
    allowed_extensions = ['.obj', '.ply', '.stl', '.off', '.dae', '.gltf', '.glb']
    mesh_files = []
    abs_directory_path = os.path.abspath(directory)
    if not os.path.isdir(abs_directory_path):
        print(f"Error: Directory '{abs_directory_path}' not found.")
        print(f"Please ensure MESH_DIR = \"{directory}\" is correctly set.")
        return []
    
    print(f"Searching for meshes in: {abs_directory_path}")
    for f_name in os.listdir(abs_directory_path):
        if os.path.isfile(os.path.join(abs_directory_path, f_name)) and \
           any(f_name.lower().endswith(ext) for ext in allowed_extensions):
            mesh_files.append(f_name)
    
    if not mesh_files:
        print(f"No mesh files found in {abs_directory_path}.")
    else:
        print(f"Found meshes: {mesh_files}")
    return mesh_files

# --- Volume Estimation Methods ---
def estimate_volume_voxelization(mesh, mesh_name, resolution_divisions):
    """
    Estimates volume using voxelization.
    resolution_divisions: number of voxels along the longest dimension.
    """
    print(f"\n--- Voxelization for {mesh_name} (Target divisions: {resolution_divisions}) ---")
    
    if not mesh.is_watertight:
        print(f"Warning: Mesh {mesh_name} is not watertight. Voxelization results may be less accurate.")
        # Optionally, you could try to fill holes, but this alters the mesh:
        # print("Attempting to fill holes...")
        # mesh.fill_holes()
        # if not mesh.is_watertight:
        #     print("Mesh is still not watertight after attempting to fill holes.")

    start_time = time.time()

    try:
        # Determine pitch based on resolution_divisions and mesh extent
        if mesh.extents.max() == 0: # Handle degenerate meshes
            print("Error: Mesh has zero extent. Cannot voxelize.")
            return None, None, None, None
        
        pitch = mesh.extents.max() / resolution_divisions
        
        # Voxelize the mesh
        voxel_grid = mesh.voxelized(pitch=pitch)
        
        if not isinstance(voxel_grid, trimesh.voxel.VoxelGrid):
             print(f"Failed to voxelize mesh {mesh_name}. Result was not a VoxelGrid.")
             return None, None, time.time() - start_time, None

        filled_count = voxel_grid.filled_count
        volume_per_voxel = pitch ** 3
        estimated_volume = filled_count * volume_per_voxel
        
        actual_resolution = voxel_grid.shape # (x_voxels, y_voxels, z_voxels)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"  Actual Voxel Grid Shape: {actual_resolution}")
        print(f"  Calculated Pitch: {pitch:.6f}")
        print(f"  Filled Voxels: {filled_count}")
        print(f"  Estimated Volume: {estimated_volume:.6f}")
        print(f"  Computation Time: {computation_time:.4f} seconds")
        
        return estimated_volume, computation_time, pitch, actual_resolution

    except Exception as e:
        print(f"Error during voxelization of {mesh_name} with target divisions {resolution_divisions}: {e}")
        return None, None, time.time() - start_time, None


def estimate_volume_sdf_monte_carlo(mesh, mesh_name, num_samples):
    """
    Estimates volume using SDF and Monte Carlo sampling.
    """
    print(f"\n--- SDF + Monte Carlo for {mesh_name} ({num_samples} samples) ---")

    # SDF computation often relies on the mesh being watertight for best results.
    # if not mesh.is_watertight:
    #     print(f"Warning: Mesh {mesh_name} is not watertight. SDF accuracy might be affected.")

    start_time = time.time()
    
    try:
        # 1. Get bounding box and its volume
        min_bound, max_bound = mesh.bounds
        bbox_dims = max_bound - min_bound
        bbox_volume = np.prod(bbox_dims)

        if bbox_volume <= 1e-9: # Check for degenerate bounding box
            print(f"Error: Bounding box of {mesh_name} has zero or very small volume ({bbox_volume:.2e}). Cannot sample.")
            return None, None, None, None, None

        # 2. Randomly sample points in the mesh's bounding box
        # Generates points in [0,1)^3 then scales and shifts them
        random_points = np.random.random((num_samples, 3)) 
        random_points = random_points * bbox_dims + min_bound

        # 3. Compute SDF and classify points
        # ProximityQuery can be slow to initialize for complex meshes, but fast for queries.
        proximity_query_init_start = time.time()
        query_engine = trimesh.proximity.ProximityQuery(mesh)
        proximity_query_init_time = time.time() - proximity_query_init_start
        
        signed_distances_start = time.time()
        signed_distances = query_engine.signed_distance(random_points)
        signed_distances_time = time.time() - signed_distances_start
        
        num_inside = np.sum(signed_distances <= 0) # Points on surface (SDF=0 or SDF<0) are inside

        # 4. Estimate Volume
        fraction_inside = num_inside / num_samples
        estimated_volume = fraction_inside * bbox_volume
        
        end_time = time.time()
        computation_time = end_time - start_time

        print(f"  Bounding Box Volume: {bbox_volume:.6f}")
        print(f"  Points Inside / Total Samples: {num_inside} / {num_samples}")
        print(f"  Fraction Inside: {fraction_inside:.6f}")
        print(f"  Estimated Volume: {estimated_volume:.6f}")
        print(f"  ProximityQuery Init Time: {proximity_query_init_time:.4f}s")
        print(f"  Signed Distance Query Time: {signed_distances_time:.4f}s")
        print(f"  Total Computation Time: {computation_time:.4f} seconds")
        
        return estimated_volume, computation_time, bbox_volume, num_inside, num_samples

    except Exception as e:
        print(f"Error during SDF+MC of {mesh_name} with {num_samples} samples: {e}")
        return None, None, time.time() - start_time, None, None

# --- Main Execution Logic ---
def main():
    mesh_file_names = get_mesh_files(MESH_DIR)
    if not mesh_file_names:
        return

    all_results = {} 

    for mesh_file_name in mesh_file_names:
        mesh_path = os.path.join(MESH_DIR, mesh_file_name)
        print(f"\n=====================================================")
        print(f"Processing Mesh: {mesh_file_name} (Path: {mesh_path})")
        print(f"=====================================================")
        
        try:
            # Load mesh once per file
            load_start_time = time.time()
            mesh = trimesh.load_mesh(mesh_path, process=True) # process=True is important
            load_time = time.time() - load_start_time
            print(f"Mesh loaded in {load_time:.4f}s. Watertight: {mesh.is_watertight}, Volume (trimesh built-in): {mesh.volume:.6f}")
            
            if mesh.is_empty:
                print(f"Mesh {mesh_file_name} is empty or could not be loaded properly. Skipping.")
                continue
        except Exception as e:
            print(f"Failed to load mesh {mesh_file_name}: {e}")
            continue

        mesh_results_key = os.path.basename(mesh_path)
        all_results[mesh_results_key] = {"voxelization": [], "sdf_monte_carlo": [], "trimesh_volume": mesh.volume}

        print("\n>>> Starting Voxelization Tests...")
        for res_div in VOXEL_RESOLUTIONS_DIVISIONS:
            volume, comp_time, pitch, actual_res = estimate_volume_voxelization(mesh, mesh_results_key, res_div)
            if volume is not None:
                all_results[mesh_results_key]["voxelization"].append({
                    "target_divisions": res_div,
                    "pitch": pitch,
                    "actual_resolution": actual_res,
                    "volume": volume,
                    "time": comp_time
                })

        print("\n>>> Starting SDF + Monte Carlo Tests...")
        for n_samples in MONTE_CARLO_SAMPLE_COUNTS:
            volume, comp_time, bbox_vol, n_in, n_tot = estimate_volume_sdf_monte_carlo(mesh, mesh_results_key, n_samples)
            if volume is not None:
                all_results[mesh_results_key]["sdf_monte_carlo"].append({
                    "samples": n_samples,
                    "points_inside": n_in,
                    "bbox_volume": bbox_vol,
                    "volume": volume,
                    "time": comp_time
                })
    
    # --- Reporting Results ---
    print("\n\n--- Summary of All Results ---")
    for mesh_file, data in all_results.items():
        print(f"\nMesh: {mesh_file} (Trimesh reference volume: {data.get('trimesh_volume', 'N/A'):.6f})")
        
        print("  Voxelization Results:")
        if data["voxelization"]:
            for res_data in data["voxelization"]:
                print(f"    Target Divisions: {res_data['target_divisions']:<4} "
                      f"(Actual Res: {str(res_data['actual_resolution']):<18}, Pitch: {res_data['pitch']:.5f}): "
                      f"Volume = {res_data['volume']:.6f}, Time = {res_data['time']:.4f}s")
        else:
            print("    No successful voxelization runs.")

        print("  SDF + Monte Carlo Results:")
        if data["sdf_monte_carlo"]:
            for mc_data in data["sdf_monte_carlo"]:
                print(f"    Samples: {mc_data['samples']:<7} "
                      f"(Points Inside: {mc_data['points_inside']!s:<7}, BBox Vol: {mc_data['bbox_volume']:.4f}): "
                      f"Volume = {mc_data['volume']:.6f}, Time = {mc_data['time']:.4f}s")
        else:
            print("    No successful SDF + Monte Carlo runs.")
            
    # For your report, you would typically copy these results into a table or use them
    # to generate plots (e.g., using matplotlib) to show convergence, accuracy vs. trimesh.volume,
    # and performance trade-offs.

if __name__ == "__main__":
    main()
