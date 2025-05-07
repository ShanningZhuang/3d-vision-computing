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
MONTE_CARLO_SAMPLE_COUNTS = [1000, 2000, 5000, 10000] # Reduced 1M for quicker runs

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
        
        # Voxelize the mesh
        voxel_grid = mesh.voxelized(pitch=pitch)

        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB
        
        if not isinstance(voxel_grid, trimesh.voxel.VoxelGrid):
             print(f"Failed to voxelize mesh {mesh_name}. Result was not a VoxelGrid.")
             return None, None, time.time() - start_time, None, current_rss_mb

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
        if current_rss_mb is not None:
            print(f"  Memory RSS after voxelization: {current_rss_mb:.2f} MB")
        print(f"  Computation Time: {computation_time:.4f} seconds")
        
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
        signed_distances_start = time.time()
        signed_distances = trimesh.proximity.signed_distance(mesh, random_points)
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
