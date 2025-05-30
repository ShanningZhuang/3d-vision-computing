import trimesh
import numpy as np
import time
import os

# Add visualization imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("matplotlib library found. Will generate voxelization plots.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib library not found. Voxelization plots will not be generated.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("open3d library found. Will generate 3D visualizations.")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("open3d library not found. 3D visualizations will not be generated.")

# --- Configuration ---
# Adjust MESH_DIR if your 'objs_approx' folder is located elsewhere relative to this script.
# Assumes this script is in a 'my_solution' type folder, and 'objs_approx' is a sibling to its parent.
# e.g., script is in '.../Homework1/my_solution/' and meshes are in '.../Homework1/objs_approx/'
MESH_DIR = "../../objs_approx"

# Resolutions for Voxelization: number of divisions along the mesh's longest dimension.
# Higher numbers mean finer resolution.
VOXEL_RESOLUTIONS_DIVISIONS = [32, 64, 128, 192] # Reduced 256 due to potential memory/time

# Number of samples for SDF + Monte Carlo.
MONTE_CARLO_SAMPLE_COUNTS = [1000, 2000, 5000, 10000] # Reduced 1M for quicker runs

# Visualization settings
ENABLE_VOXEL_VISUALIZATION = True  # Set to False to disable visualization
VISUALIZATION_OUTPUT_DIR = "./voxelization_debug"

# Attempt to import psutil for memory tracking
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil library found. Will attempt to record memory usage.")
except ImportError:
    print("psutil library not found. Memory usage will not be recorded by the script.")
    print("For memory profiling, consider installing psutil (`pip install psutil`)")
    print("or using a dedicated profiler like memory_profiler.")

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

# --- Visualization Functions ---
def create_output_directory():
    """Create output directory for visualizations."""
    if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
        os.makedirs(VISUALIZATION_OUTPUT_DIR)
        print(f"Created visualization output directory: {VISUALIZATION_OUTPUT_DIR}")

def visualize_voxel_grid_matplotlib(voxel_grid, mesh_name, resolution_divisions, pitch):
    """Visualize voxel grid using matplotlib."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_VOXEL_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        # Get filled voxel coordinates
        filled_coords = np.array(np.where(voxel_grid.matrix)).T
        
        if len(filled_coords) == 0:
            print(f"Warning: No filled voxels found for {mesh_name} at resolution {resolution_divisions}")
            return
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot filled voxels
        ax.scatter(filled_coords[:, 0], filled_coords[:, 1], filled_coords[:, 2], 
                  c='red', alpha=0.6, s=1, label=f'Filled Voxels ({len(filled_coords)})')
        
        # Set labels and title
        ax.set_xlabel('X Voxels')
        ax.set_ylabel('Y Voxels')
        ax.set_zlabel('Z Voxels')
        ax.set_title(f'Voxelization of {mesh_name}\nResolution: {resolution_divisions}, Pitch: {pitch:.6f}')
        ax.legend()
        
        # Save plot
        output_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                 f"{mesh_name}_voxels_res{resolution_divisions}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved voxel visualization: {output_file}")
        plt.close()
        
        # Create cross-section views
        create_voxel_cross_sections(voxel_grid, mesh_name, resolution_divisions, pitch)
        
    except Exception as e:
        print(f"Error creating matplotlib visualization for {mesh_name}: {e}")

def create_voxel_cross_sections(voxel_grid, mesh_name, resolution_divisions, pitch):
    """Create cross-section views of the voxel grid."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        matrix = voxel_grid.matrix
        shape = matrix.shape
        
        # Create cross-sections at the middle of each dimension
        mid_x, mid_y, mid_z = shape[0]//2, shape[1]//2, shape[2]//2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY plane (Z = mid_z)
        xy_slice = matrix[:, :, mid_z]
        axes[0].imshow(xy_slice.T, origin='lower', cmap='Reds', alpha=0.8)
        axes[0].set_title(f'XY Cross-section (Z={mid_z})')
        axes[0].set_xlabel('X Voxels')
        axes[0].set_ylabel('Y Voxels')
        
        # XZ plane (Y = mid_y)
        xz_slice = matrix[:, mid_y, :]
        axes[1].imshow(xz_slice.T, origin='lower', cmap='Reds', alpha=0.8)
        axes[1].set_title(f'XZ Cross-section (Y={mid_y})')
        axes[1].set_xlabel('X Voxels')
        axes[1].set_ylabel('Z Voxels')
        
        # YZ plane (X = mid_x)
        yz_slice = matrix[mid_x, :, :]
        axes[2].imshow(yz_slice.T, origin='lower', cmap='Reds', alpha=0.8)
        axes[2].set_title(f'YZ Cross-section (X={mid_x})')
        axes[2].set_xlabel('Y Voxels')
        axes[2].set_ylabel('Z Voxels')
        
        plt.suptitle(f'Cross-sections of {mesh_name} (Res: {resolution_divisions})')
        plt.tight_layout()
        
        output_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                 f"{mesh_name}_cross_sections_res{resolution_divisions}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved cross-section visualization: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating cross-section visualization: {e}")

def visualize_mesh_and_voxels_open3d(mesh, voxel_grid, mesh_name, resolution_divisions):
    """Visualize original mesh and voxel grid together using Open3D."""
    if not OPEN3D_AVAILABLE or not ENABLE_VOXEL_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        # Convert trimesh to open3d mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
        o3d_mesh.compute_vertex_normals()
        
        # Convert voxel grid to point cloud
        filled_coords = np.array(np.where(voxel_grid.matrix)).T
        
        if len(filled_coords) == 0:
            print(f"Warning: No filled voxels to visualize for {mesh_name}")
            return
        
        # Convert voxel indices to world coordinates
        # Each voxel center is at: origin + (index + 0.5) * pitch
        voxel_centers = []
        for coord in filled_coords:
            # Convert from voxel indices to world coordinates
            world_coord = voxel_grid.origin + (coord + 0.5) * voxel_grid.pitch
            voxel_centers.append(world_coord)
        
        voxel_points = np.array(voxel_centers)
        
        # Create point cloud for voxels
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
        
        # Save visualization
        output_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                 f"{mesh_name}_mesh_and_voxels_res{resolution_divisions}.ply")
        
        # Combine geometries and save
        combined_geometry = [o3d_mesh, voxel_pcd]
        
        print(f"  Open3D visualization ready for {mesh_name} (Res: {resolution_divisions})")
        print(f"  Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Voxel points: {len(voxel_points)} filled voxels")
        
        # Optionally save the point cloud
        o3d.io.write_point_cloud(output_file.replace('.ply', '_voxels.ply'), voxel_pcd)
        print(f"  Saved voxel point cloud: {output_file.replace('.ply', '_voxels.ply')}")
        
    except Exception as e:
        print(f"Error creating Open3D visualization for {mesh_name}: {e}")

def analyze_voxelization_quality(mesh, voxel_grid, mesh_name, resolution_divisions, pitch):
    """Analyze the quality of voxelization and print diagnostic information."""
    print(f"\n  === Voxelization Quality Analysis for {mesh_name} ===")
    
    # Basic voxel grid info
    print(f"  Voxel Grid Shape: {voxel_grid.shape}")
    print(f"  Total Voxels: {np.prod(voxel_grid.shape):,}")
    print(f"  Filled Voxels: {voxel_grid.filled_count:,}")
    print(f"  Fill Ratio: {voxel_grid.filled_count / np.prod(voxel_grid.shape):.4f}")
    
    # Mesh properties
    print(f"  Original Mesh Bounds: {mesh.bounds}")
    print(f"  Mesh Extents: {mesh.extents}")
    print(f"  Mesh Volume (trimesh): {mesh.volume:.6f}")
    print(f"  Mesh Surface Area: {mesh.area:.6f}")
    print(f"  Mesh is Watertight: {mesh.is_watertight}")
    
    # Voxel grid properties
    print(f"  Voxel Grid Bounds: {voxel_grid.bounds}")
    print(f"  Voxel Pitch: {pitch:.6f}")
    print(f"  Voxel Volume: {pitch**3:.8f}")
    
    # Volume comparison
    estimated_volume = voxel_grid.filled_count * (pitch ** 3)
    if mesh.volume > 0:
        volume_error = abs(estimated_volume - mesh.volume) / mesh.volume * 100
        print(f"  Volume Error: {volume_error:.2f}%")
    
    # Check for potential issues
    if voxel_grid.filled_count == 0:
        print("  WARNING: No voxels are filled! This suggests a problem with voxelization.")
    elif voxel_grid.filled_count == np.prod(voxel_grid.shape):
        print("  WARNING: All voxels are filled! This might indicate incorrect voxelization.")
    
    if not mesh.is_watertight:
        print("  NOTE: Mesh is not watertight, which may affect voxelization accuracy.")

# --- Volume Estimation Methods ---
def estimate_volume_voxelization(mesh, mesh_name, resolution_divisions):
    """
    Estimates volume using voxelization.
    resolution_divisions: number of voxels along the longest dimension.
    """
    print(f"\n--- Voxelization for {mesh_name} (Target divisions: {resolution_divisions}) ---")
    
    current_rss_mb = None # Initialize memory usage variable

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
            return None, None, None, None, None # Added None for memory
        
        pitch = mesh.extents.max() / resolution_divisions
        
        # Voxelize the mesh (this creates surface voxelization)
        voxel_grid = mesh.voxelized(pitch=pitch)

        if not isinstance(voxel_grid, trimesh.voxel.VoxelGrid):
             print(f"Failed to voxelize mesh {mesh_name}. Result was not a VoxelGrid.")
             return None, None, time.time() - start_time, None, current_rss_mb

        # Get surface voxel count for comparison
        surface_filled_count = voxel_grid.filled_count
        print(f"  Surface Voxels (before fill): {surface_filled_count}")
        
        # IMPORTANT: Fill the voxel grid to create a solid volume
        # This converts the surface voxelization to a solid, filled voxelization
        voxel_grid.fill()
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB

        filled_count = voxel_grid.filled_count
        volume_per_voxel = pitch ** 3
        estimated_volume = filled_count * volume_per_voxel
        
        actual_resolution = voxel_grid.shape # (x_voxels, y_voxels, z_voxels)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"  Actual Voxel Grid Shape: {actual_resolution}")
        print(f"  Calculated Pitch: {pitch:.6f}")
        print(f"  Filled Voxels (after fill): {filled_count}")
        print(f"  Fill Ratio Increase: {filled_count / surface_filled_count:.2f}x" if surface_filled_count > 0 else "")
        print(f"  Estimated Volume: {estimated_volume:.6f}")
        if current_rss_mb is not None:
            print(f"  Memory RSS after voxelization: {current_rss_mb:.2f} MB")
        print(f"  Computation Time: {computation_time:.4f} seconds")
        
        # Add visualization and analysis
        if ENABLE_VOXEL_VISUALIZATION:
            analyze_voxelization_quality(mesh, voxel_grid, mesh_name, resolution_divisions, pitch)
            visualize_voxel_grid_matplotlib(voxel_grid, mesh_name, resolution_divisions, pitch)
            visualize_mesh_and_voxels_open3d(mesh, voxel_grid, mesh_name, resolution_divisions)
        
        return estimated_volume, computation_time, pitch, actual_resolution, current_rss_mb

    except Exception as e:
        print(f"Error during voxelization of {mesh_name} with target divisions {resolution_divisions}: {e}")
        if PSUTIL_AVAILABLE: # Still try to get memory if error happens after psutil could be called
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return None, None, time.time() - start_time, None, current_rss_mb


def estimate_volume_sdf_monte_carlo(mesh, mesh_name, num_samples):
    """
    Estimates volume using SDF and Monte Carlo sampling using trimesh.proximity.signed_distance directly.
    """
    print(f"\n--- SDF + Monte Carlo for {mesh_name} ({num_samples} samples) using direct signed_distance ---")
    current_rss_mb = None # Initialize memory usage variable
    start_time = time.time()
    
    try:
        # 1. Get bounding box and its volume
        min_bound, max_bound = mesh.bounds
        bbox_dims = max_bound - min_bound
        bbox_volume = np.prod(bbox_dims)

        if bbox_volume <= 1e-9: # Check for degenerate bounding box
            print(f"Error: Bounding box of {mesh_name} has zero or very small volume ({bbox_volume:.2e}). Cannot sample.")
            return None, None, None, None, None, None # Added None for memory

        # 2. Randomly sample points in the mesh's bounding box
        random_points = np.random.random((num_samples, 3)) 
        random_points = random_points * bbox_dims + min_bound

        # 3. Compute SDF and classify points using trimesh.proximity.signed_distance
        # Process in batches to reduce memory usage
        batch_size = 1000
        signed_distances_start = time.time()
        
        # Initialize array to store all signed distances
        signed_distances = np.zeros(num_samples)
        
        # Process points in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_points = random_points[i:end_idx]
            
            # Compute signed distances for this batch
            batch_signed_distances = trimesh.proximity.signed_distance(mesh, batch_points)
            signed_distances[i:end_idx] = batch_signed_distances
            
            # Optional: Print progress for large datasets
            if num_samples > 10000 and (i // batch_size) % 10 == 0:
                progress = (end_idx / num_samples) * 100
                print(f"    Batch progress: {progress:.1f}% ({end_idx}/{num_samples} points)")
        
        signed_distances_time = time.time() - signed_distances_start

        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB
        
        num_inside = np.sum(signed_distances >= 0) 

        # 4. Estimate Volume
        fraction_inside = num_inside / num_samples
        estimated_volume = fraction_inside * bbox_volume
        
        end_time = time.time()
        computation_time = end_time - start_time

        print(f"  Bounding Box Volume: {bbox_volume:.6f}")
        print(f"  Points Inside / Total Samples: {num_inside} / {num_samples}")
        print(f"  Fraction Inside: {fraction_inside:.6f}")
        print(f"  Estimated Volume: {estimated_volume:.6f}")
        if current_rss_mb is not None:
            print(f"  Memory RSS after SDF calculation: {current_rss_mb:.2f} MB")
        print(f"  Direct Signed Distance Query Time: {signed_distances_time:.4f}s")
        print(f"  Total Computation Time: {computation_time:.4f} seconds")
        
        return estimated_volume, computation_time, bbox_volume, num_inside, num_samples, current_rss_mb

    except Exception as e:
        print(f"Error during SDF+MC (direct) of {mesh_name} with {num_samples} samples: {e}")
        if PSUTIL_AVAILABLE: # Still try to get memory
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return None, None, time.time() - start_time, None, None, current_rss_mb

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
            print(f"Mesh loaded in {load_time:.4f}s. Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}, Watertight: {mesh.is_watertight}, Volume (trimesh built-in): {mesh.volume:.6f}")
            
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
            volume, comp_time, pitch, actual_res, memory_rss = estimate_volume_voxelization(mesh, mesh_results_key, res_div)
            if volume is not None:
                all_results[mesh_results_key]["voxelization"].append({
                    "target_divisions": res_div,
                    "pitch": pitch,
                    "actual_resolution": actual_res,
                    "volume": volume,
                    "time": comp_time,
                    "memory_rss_mb": memory_rss
                })

        print("\n>>> Starting SDF + Monte Carlo Tests (using direct signed_distance)...")
        
        for n_samples in MONTE_CARLO_SAMPLE_COUNTS:
            volume, comp_time, bbox_vol, n_in, n_tot, memory_rss = estimate_volume_sdf_monte_carlo(mesh, mesh_results_key, n_samples)
            if volume is not None:
                all_results[mesh_results_key]["sdf_monte_carlo"].append({
                    "samples": n_samples,
                    "points_inside": n_in,
                    "bbox_volume": bbox_vol,
                    "volume": volume,
                    "time": comp_time,
                    "memory_rss_mb": memory_rss
                })
    
    # --- Reporting Results ---
    print("\n\n--- Summary of All Results ---")
    for mesh_file, data in all_results.items():
        print(f"\nMesh: {mesh_file} (Trimesh reference volume: {data.get('trimesh_volume', 'N/A'):.6f})")
        
        print("  Voxelization Results:")
        if data["voxelization"]:
            for res_data in data["voxelization"]:
                mem_str = f", Mem: {res_data['memory_rss_mb']:.2f} MB" if res_data.get('memory_rss_mb') is not None else ""
                print(f"    Target Divisions: {res_data['target_divisions']:<4} "
                      f"(Actual Res: {str(res_data['actual_resolution']):<18}, Pitch: {res_data['pitch']:.5f}): "
                      f"Volume = {res_data['volume']:.6f}, Time = {res_data['time']:.4f}s{mem_str}")
        else:
            print("    No successful voxelization runs.")

        print("  SDF + Monte Carlo Results:")
        if data["sdf_monte_carlo"]:
            for mc_data in data["sdf_monte_carlo"]:
                mem_str = f", Mem: {mc_data['memory_rss_mb']:.2f} MB" if mc_data.get('memory_rss_mb') is not None else ""
                print(f"    Samples: {mc_data['samples']:<7} "
                      f"(Points Inside: {str(mc_data['points_inside']):<7}, BBox Vol: {mc_data['bbox_volume']:.4f}): "
                      f"Volume = {mc_data['volume']:.6f}, Time = {mc_data['time']:.4f}s{mem_str}")
        else:
            print("    No successful SDF + Monte Carlo runs.")
            
    # For your report, you would typically copy these results into a table or use them
    # to generate plots (e.g., using matplotlib) to show convergence, accuracy vs. trimesh.volume,
    # and performance trade-offs.

    generate_markdown_report(all_results)

# --- Function to Generate Markdown Report ---
def generate_markdown_report(all_results, report_filename="volume_estimation_report.md"):
    report_parts = []

    # --- Title and Intro ---
    report_parts.append("# Volume Estimation Method Comparison\n\n")
    report_parts.append("This report compares two methods for estimating the volume of 3D meshes: Voxelization and Signed Distance Field (SDF) + Monte Carlo. ")
    report_parts.append("The comparison focuses on accuracy (compared to `trimesh` built-in volume calculation), speed (computation time), and a qualitative discussion of memory usage.\n\n")

    report_parts.append("## Methodology\n\n")
    report_parts.append("- **Meshes**: Processed from the `MESH_DIR` directory as configured in the script.\n")
    report_parts.append("- **Ground Truth**: `trimesh.Trimesh.volume` property is used as the reference volume for accuracy comparison.\n")
    report_parts.append("- **Voxelization**: Meshes are voxelized at different target divisions along the mesh's longest dimension. Volume is calculated as `filled_voxels * pitch^3`.\n")
    report_parts.append("- **SDF + Monte Carlo**: Points are randomly sampled within the mesh's bounding box. The Signed Distance Field (SDF) is used to classify points as inside or outside the mesh. Volume is estimated as `(points_inside / total_samples) * bounding_box_volume`.\n")
    report_parts.append("- **Accuracy Metric**: Percentage difference from Trimesh volume: `((Estimated Volume - Trimesh Volume) / Trimesh Volume) * 100%`. Values are rounded to two decimal places.\n")

    report_parts.append("\n## Results per Mesh\n")

    for mesh_file, data in all_results.items():
        trimesh_volume = data.get('trimesh_volume') # mesh.volume should be a float
        
        report_parts.append(f"\n### Mesh: {mesh_file}\n")
        if trimesh_volume is not None:
             report_parts.append(f"**Trimesh Reference Volume:** `{trimesh_volume:.6f}`\n")
        else:
            report_parts.append(f"**Trimesh Reference Volume:** N/A (Could not be determined or mesh was empty)\n")

        # Voxelization Results
        report_parts.append("\n#### Voxelization Results\n")
        if data.get("voxelization"):
            header = "| Target Divisions | Actual Resolution   | Pitch   | Estimated Volume | % Diff from Trimesh | Time (s) |"
            separator = "|------------------|---------------------|---------|------------------|---------------------|----------|"
            if PSUTIL_AVAILABLE and data["voxelization"] and data["voxelization"][0].get("memory_rss_mb") is not None:
                header += " Memory RSS (MB) |"
                separator += "-----------------|"
            report_parts.append(header + "\n")
            report_parts.append(separator + "\n")

            for res_data in data["voxelization"]:
                est_vol = res_data['volume']
                perc_diff_str = "N/A"
                if trimesh_volume is not None and trimesh_volume != 0:
                    perc_diff = ((est_vol - trimesh_volume) / trimesh_volume) * 100
                    perc_diff_str = f"{perc_diff:.2f}%"
                elif trimesh_volume is not None and trimesh_volume == 0:
                    perc_diff_str = "Inf" if est_vol != 0 else "0.00%"
                
                row = f"| {res_data['target_divisions']:<16} | {str(res_data['actual_resolution']):<19} | {res_data['pitch']:.5f} | {est_vol:.6f}         | {perc_diff_str:<19} | {res_data['time']:.4f}   |"
                if PSUTIL_AVAILABLE and res_data.get("memory_rss_mb") is not None:
                    row += f" {res_data['memory_rss_mb']:.2f}            |"
                report_parts.append(row + "\n")
        else:
            report_parts.append("No successful voxelization runs or data not available.\n")

        # SDF + Monte Carlo Results
        report_parts.append("\n#### SDF + Monte Carlo Results (using trimesh.proximity.signed_distance)\n") # Updated sub-header
        if data.get("sdf_monte_carlo"):
            header = "| Sample Count | Points Inside   | BBox Volume | Estimated Volume | % Diff from Trimesh | Time (s) |"
            separator = "|--------------|-----------------|-------------|------------------|---------------------|----------|"
            if PSUTIL_AVAILABLE and data["sdf_monte_carlo"] and data["sdf_monte_carlo"][0].get("memory_rss_mb") is not None:
                header += " Memory RSS (MB) |"
                separator += "-----------------|"
            report_parts.append(header + "\n")
            report_parts.append(separator + "\n")
            
            for mc_data in data["sdf_monte_carlo"]:
                est_vol = mc_data['volume']
                perc_diff_str = "N/A"
                if trimesh_volume is not None and trimesh_volume != 0:
                    perc_diff = ((est_vol - trimesh_volume) / trimesh_volume) * 100
                    perc_diff_str = f"{perc_diff:.2f}%"
                elif trimesh_volume is not None and trimesh_volume == 0:
                     perc_diff_str = "Inf" if est_vol != 0 else "0.00%"

                time_str = f"{mc_data['time']:.4f}"
                row = f"| {mc_data['samples']:<12} | {str(mc_data['points_inside']):<15} | {mc_data['bbox_volume']:.4f}    | {est_vol:.6f}         | {perc_diff_str:<19} | {time_str:<8} |" # Adjusted time_str padding
                if PSUTIL_AVAILABLE and mc_data.get("memory_rss_mb") is not None:
                    row += f" {mc_data['memory_rss_mb']:.2f}            |"
                report_parts.append(row + "\n")
        else:
            report_parts.append("No successful SDF + Monte Carlo runs or data not available.\n")
        
        report_parts.append("\n**Brief Analysis for this Mesh (to be filled in by the user):**\n")
        report_parts.append("- *Accuracy observations (e.g., convergence with parameters, comparison between methods for this mesh):*\n")
        report_parts.append("- *Speed observations (e.g., how time scales with parameters, which method was faster for this mesh):*\n")
        report_parts.append("- *Any notable issues or behaviors for this specific mesh (e.g., impact of being non-watertight if applicable):*\n")

    # --- Overall Discussion ---
    report_parts.append("\n\n## Overall Comparison and Discussion\n")
    report_parts.append("*(This section should be completed by the user after reviewing the results across all meshes.)*\n")
    report_parts.append("\n### Accuracy:\n")
    report_parts.append("- *Discuss general trends for Voxelization accuracy as resolution changes. Does it consistently improve? What are the limits?*\n")
    report_parts.append("- *Discuss general trends for SDF + Monte Carlo accuracy as the number of samples changes (using direct `trimesh.proximity.signed_distance`). How does it converge?*\n")
    report_parts.append("- *Compare the overall accuracy: Which method tended to be more accurate? Were there types of meshes or scenarios where one outperformed the other?*\n")
    report_parts.append("- *Consider the impact of mesh properties (e.g., watertightness, complexity) on accuracy for both methods.*\n")

    report_parts.append("\n### Speed:\n")
    report_parts.append("- *Discuss general trends for Voxelization speed as resolution increases.*\n")
    report_parts.append("- *Discuss general trends for SDF + Monte Carlo speed (using direct `trimesh.proximity.signed_distance`) as the number of samples increases. Note that each call to `signed_distance` might perform its own setup, potentially impacting performance for many small batches compared to a pre-initialized `ProximityQuery`.* \n")
    report_parts.append("- *Compare overall speed: Which method was generally faster for comparable levels of accuracy or effort?*\n")

    report_parts.append("\n### Memory Usage (Quantitative and Qualitative):\n") # Updated section title
    if PSUTIL_AVAILABLE:
        report_parts.append("The script recorded Resident Set Size (RSS) memory using `psutil` after the core computation of each method. These values are reported in the tables above (in MB) and give an indication of the process memory footprint.\n\n")
    else:
        report_parts.append("The script could not record quantitative memory usage as the `psutil` library was not found. The following discussion is qualitative.\n\n")
    
    report_parts.append("The script does not quantitatively measure memory usage during execution. However, qualitative observations can be made:\n")
    report_parts.append("- **Voxelization**: Memory usage is primarily determined by the size of the voxel grid (number of voxels along each dimension). It scales with `width * height * depth`. High resolutions lead to a cubic increase in memory, which can be substantial and a limiting factor.\n")
    report_parts.append("- **SDF + Monte Carlo (using `trimesh.proximity.signed_distance`)**:\n")
    report_parts.append("  - *SDF Calculation*: Memory is used during the call to `trimesh.proximity.signed_distance`. While it doesn't maintain a persistent large acceleration structure like `ProximityQuery`, the internal operations for each call will consume memory based on mesh complexity and the number of query points. This approach avoids the potentially large, persistent memory footprint of a `ProximityQuery` object.\n")
    report_parts.append("  - *Monte Carlo Sampling*: Storing the sample points and their signed distances requires memory proportional to the number of samples. This is generally less demanding than high-resolution voxel grids for typical sample counts (e.g., 10k-1M points).\n")
    report_parts.append("For precise memory profiling, external tools or libraries (e.g., Python's `memory_profiler`, `psutil`, or system-specific monitoring tools) would need to be employed during script execution.\n")

    report_parts.append("\n### Conclusions and Trade-offs:\n")
    report_parts.append("- *Summarize the key findings from your experiments regarding accuracy, speed, and perceived memory demands.*\n")
    report_parts.append("- *Discuss the trade-offs: For example, Voxelization might be faster at low resolutions but less accurate and very memory-hungry at high resolutions. SDF+MC might offer better accuracy for complex shapes if enough samples are used, but SDF computation can be slow/memory-intensive for highly complex meshes.*\n")
    report_parts.append("- *Provide recommendations: When would you choose Voxelization? When would SDF + Monte Carlo be preferred? Consider factors like required accuracy, available computation time, memory constraints, and mesh characteristics.*\n")
    report_parts.append("- *Mention any limitations of your study or challenges encountered (e.g., choice of parameters, impact of non-watertight meshes on results for both methods, range of meshes tested).*\n")

    # --- Write to File ---
    try:
        output_path = os.path.join(os.path.dirname(__file__) or '.', report_filename)
        with open(output_path, "w") as f:
            f.write("".join(report_parts))
        print(f"\nMarkdown report generated: {os.path.abspath(output_path)}")
    except IOError as e:
        print(f"Error writing markdown report to {report_filename}: {e}")


if __name__ == "__main__":
    main()

# --- Debug/Test Function ---
def test_single_mesh_voxelization(mesh_filename="bunny.obj", target_resolution=32):
    """
    Test voxelization on a single mesh with visualization for debugging.
    This is useful for quickly checking voxelization issues.
    """
    print(f"\n=== DEBUGGING VOXELIZATION FOR {mesh_filename} ===")
    
    mesh_path = os.path.join(MESH_DIR, mesh_filename)
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file {mesh_path} not found.")
        available_files = get_mesh_files(MESH_DIR)
        if available_files:
            print(f"Available files: {available_files}")
        return
    
    try:
        # Load mesh
        print(f"Loading mesh: {mesh_path}")
        mesh = trimesh.load_mesh(mesh_path, process=True)
        print(f"Mesh loaded successfully:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Volume (trimesh): {mesh.volume:.6f}")
        print(f"  Bounds: {mesh.bounds}")
        print(f"  Extents: {mesh.extents}")
        
        # Test voxelization with visualization enabled
        global ENABLE_VOXEL_VISUALIZATION
        ENABLE_VOXEL_VISUALIZATION = True
        
        volume, comp_time, pitch, actual_res, memory_rss = estimate_volume_voxelization(
            mesh, mesh_filename, target_resolution
        )
        
        if volume is not None:
            print(f"\nVoxelization completed successfully!")
            print(f"Estimated volume: {volume:.6f}")
            print(f"Volume error: {abs(volume - mesh.volume) / mesh.volume * 100:.2f}%")
        else:
            print(f"\nVoxelization failed!")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

# Uncomment the line below to run a quick test on a single mesh:
# test_single_mesh_voxelization("bunny.obj", 64)
