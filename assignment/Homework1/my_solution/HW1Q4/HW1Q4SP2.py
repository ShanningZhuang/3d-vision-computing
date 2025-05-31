import trimesh
import numpy as np
import time
import os
import itertools
from scipy.spatial.transform import Rotation

# Add visualization imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("matplotlib library found. Will generate intersection plots.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib library not found. Intersection plots will not be generated.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("open3d library found. Will generate 3D visualizations.")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("open3d library not found. 3D visualizations will not be generated.")

# --- Configuration ---
MESH_DIR = "../../objs_approx"

# Parameters for intersection volume estimation
VOXEL_RESOLUTIONS = [32, 64, 128]  # Voxel divisions along longest dimension
MONTE_CARLO_SAMPLES = [1000, 5000, 10000]  # Number of MC samples

# Visualization settings
ENABLE_VISUALIZATION = True
VISUALIZATION_OUTPUT_DIR = "./intersection_debug"

# Test case configuration
TEST_MESH_PAIRS = [
    ("apple.obj", "bunny.obj"),
    ("apple.obj", "teapot.obj"),
    ("bunny.obj", "teapot.obj")
]

# Attempt to import psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil library found. Will attempt to record memory usage.")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil library not found. Memory usage will not be recorded.")

# --- Helper Functions ---
def get_mesh_files(directory):
    """Gets all common mesh files from a directory."""
    allowed_extensions = ['.obj', '.ply', '.stl', '.off', '.dae', '.gltf', '.glb']
    mesh_files = []
    abs_directory_path = os.path.abspath(directory)
    if not os.path.isdir(abs_directory_path):
        print(f"Error: Directory '{abs_directory_path}' not found.")
        return []
    
    for f_name in os.listdir(abs_directory_path):
        if os.path.isfile(os.path.join(abs_directory_path, f_name)) and \
           any(f_name.lower().endswith(ext) for ext in allowed_extensions):
            mesh_files.append(f_name)
    
    return mesh_files

def create_output_directory():
    """Create output directory for visualizations."""
    if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
        os.makedirs(VISUALIZATION_OUTPUT_DIR)
        print(f"Created visualization output directory: {VISUALIZATION_OUTPUT_DIR}")

# --- Mesh Transformation Functions ---
def apply_transform_to_mesh(mesh, translation=None, rotation=None, scale=None):
    """Apply transformation to a mesh and return a copy."""
    mesh_copy = mesh.copy()
    
    # Apply scaling first
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        mesh_copy.apply_scale(scale)
    
    # Apply rotation
    if rotation is not None:
        if isinstance(rotation, (list, tuple)) and len(rotation) == 3:
            # Euler angles in degrees
            rotation_matrix = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            # Rotation matrix
            rotation_matrix = rotation
        else:
            raise ValueError("Rotation must be either Euler angles [x,y,z] in degrees or 3x3 rotation matrix")
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        mesh_copy.apply_transform(transform_matrix)
    
    # Apply translation last
    if translation is not None:
        mesh_copy.apply_translation(translation)
    
    return mesh_copy

def generate_test_cases(mesh1, mesh2, mesh1_name, mesh2_name):
    """Generate test cases with varying intersection ratios."""
    test_cases = []
    
    # Get mesh bounds for positioning
    mesh1_bounds = mesh1.bounds
    mesh2_bounds = mesh2.bounds
    mesh1_size = mesh1.extents.max()
    mesh2_size = mesh2.extents.max()
    
    # Test Case 1: No intersection (meshes far apart)
    separation_distance = mesh1_size + mesh2_size + 1.0
    test_cases.append({
        'name': 'no_intersection',
        'description': 'Meshes separated with no overlap',
        'mesh1_transform': {'translation': [-separation_distance/2, 0, 0]},
        'mesh2_transform': {'translation': [separation_distance/2, 0, 0]},
        'expected_ratio': 0.0
    })
    
    # Test Case 2: Small intersection (~10-20%)
    small_offset = mesh1_size * 0.7
    test_cases.append({
        'name': 'small_intersection',
        'description': 'Small overlap between meshes',
        'mesh1_transform': {'translation': [-small_offset/2, 0, 0]},
        'mesh2_transform': {'translation': [small_offset/2, 0, 0]},
        'expected_ratio': 0.15
    })
    
    # Test Case 3: Medium intersection (~30-50%)
    medium_offset = mesh1_size * 0.4
    test_cases.append({
        'name': 'medium_intersection',
        'description': 'Medium overlap between meshes',
        'mesh1_transform': {'translation': [-medium_offset/2, 0, 0]},
        'mesh2_transform': {'translation': [medium_offset/2, 0, 0]},
        'expected_ratio': 0.4
    })
    
    # Test Case 4: Large intersection (~60-80%)
    large_offset = mesh1_size * 0.2
    test_cases.append({
        'name': 'large_intersection',
        'description': 'Large overlap between meshes',
        'mesh1_transform': {'translation': [-large_offset/2, 0, 0]},
        'mesh2_transform': {'translation': [large_offset/2, 0, 0]},
        'expected_ratio': 0.7
    })
    
    # Test Case 5: Maximum intersection (one mesh inside another, scaled down)
    smaller_scale = min(mesh1_size, mesh2_size) / max(mesh1_size, mesh2_size) * 0.8
    if mesh1_size > mesh2_size:
        test_cases.append({
            'name': 'max_intersection',
            'description': 'Maximum overlap - smaller mesh inside larger',
            'mesh1_transform': {'translation': [0, 0, 0]},
            'mesh2_transform': {'translation': [0, 0, 0], 'scale': smaller_scale},
            'expected_ratio': 0.9
        })
    else:
        test_cases.append({
            'name': 'max_intersection',
            'description': 'Maximum overlap - smaller mesh inside larger',
            'mesh1_transform': {'translation': [0, 0, 0], 'scale': smaller_scale},
            'mesh2_transform': {'translation': [0, 0, 0]},
            'expected_ratio': 0.9
        })
    
    # Test Case 6: Rotated intersection
    test_cases.append({
        'name': 'rotated_intersection',
        'description': 'Meshes with rotation and medium overlap',
        'mesh1_transform': {'translation': [-medium_offset/3, 0, 0], 'rotation': [0, 0, 45]},
        'mesh2_transform': {'translation': [medium_offset/3, 0, 0], 'rotation': [45, 0, 0]},
        'expected_ratio': 0.3
    })
    
    return test_cases

# --- Intersection Volume Estimation Methods ---
def estimate_intersection_volume_voxelization(mesh1, mesh2, resolution_divisions):
    """Estimate intersection volume using voxelization method."""
    print(f"  Voxelization method (resolution: {resolution_divisions})")
    
    start_time = time.time()
    current_rss_mb = None
    
    try:
        # Compute combined bounding box
        combined_bounds = np.array([
            np.minimum(mesh1.bounds[0], mesh2.bounds[0]),
            np.maximum(mesh1.bounds[1], mesh2.bounds[1])
        ])
        combined_extents = combined_bounds[1] - combined_bounds[0]
        
        if combined_extents.max() == 0:
            print("    Error: Combined mesh bounds have zero extent")
            return None, None, time.time() - start_time, None
        
        # Calculate pitch based on combined bounds
        pitch = combined_extents.max() / resolution_divisions
        
        # Voxelize both meshes with the same pitch and bounds
        print(f"    Voxelizing mesh1...")
        voxel_grid1 = mesh1.voxelized(pitch=pitch)
        voxel_grid1.fill()
        
        print(f"    Voxelizing mesh2...")
        voxel_grid2 = mesh2.voxelized(pitch=pitch)
        voxel_grid2.fill()
        
        # Ensure both voxel grids have the same coordinate system
        # We need to align them to the same origin and shape
        
        # Calculate the unified grid parameters
        unified_origin = combined_bounds[0]
        unified_shape = np.ceil(combined_extents / pitch).astype(int)
        
        # Create unified grids
        unified_grid1 = create_unified_voxel_grid(mesh1, unified_origin, pitch, unified_shape)
        unified_grid2 = create_unified_voxel_grid(mesh2, unified_origin, pitch, unified_shape)
        
        # Compute intersection
        intersection_grid = unified_grid1 & unified_grid2
        intersection_count = np.sum(intersection_grid)
        
        # Calculate volumes
        voxel_volume = pitch ** 3
        intersection_volume = intersection_count * voxel_volume
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        computation_time = time.time() - start_time
        
        print(f"    Unified grid shape: {unified_shape}")
        print(f"    Mesh1 filled voxels: {np.sum(unified_grid1)}")
        print(f"    Mesh2 filled voxels: {np.sum(unified_grid2)}")
        print(f"    Intersection voxels: {intersection_count}")
        print(f"    Intersection volume: {intersection_volume:.6f}")
        print(f"    Computation time: {computation_time:.4f}s")
        
        return intersection_volume, intersection_count, computation_time, current_rss_mb
        
    except Exception as e:
        print(f"    Error in voxelization method: {e}")
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return None, None, time.time() - start_time, current_rss_mb

def create_unified_voxel_grid(mesh, origin, pitch, shape):
    """Create a unified voxel grid for a mesh with specified origin, pitch, and shape."""
    # Create empty grid
    grid = np.zeros(shape, dtype=bool)
    
    # Voxelize the mesh
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_grid.fill()
    
    # Calculate offset between mesh voxel grid origin and unified origin
    offset = np.round((voxel_grid.origin - origin) / pitch).astype(int)
    
    # Get the bounds for copying
    mesh_shape = voxel_grid.matrix.shape
    
    # Calculate valid ranges for copying
    start_unified = np.maximum(0, offset)
    end_unified = np.minimum(shape, offset + mesh_shape)
    start_mesh = np.maximum(0, -offset)
    end_mesh = start_mesh + (end_unified - start_unified)
    
    # Copy the mesh voxel data to the unified grid
    if np.all(end_unified > start_unified) and np.all(end_mesh > start_mesh):
        grid[start_unified[0]:end_unified[0], 
             start_unified[1]:end_unified[1], 
             start_unified[2]:end_unified[2]] = \
        voxel_grid.matrix[start_mesh[0]:end_mesh[0],
                         start_mesh[1]:end_mesh[1],
                         start_mesh[2]:end_mesh[2]]
    
    return grid

def estimate_intersection_volume_monte_carlo(mesh1, mesh2, num_samples):
    """Estimate intersection volume using Monte Carlo sampling."""
    print(f"  Monte Carlo method ({num_samples} samples)")
    
    start_time = time.time()
    current_rss_mb = None
    
    try:
        # Compute combined bounding box
        combined_bounds = np.array([
            np.minimum(mesh1.bounds[0], mesh2.bounds[0]),
            np.maximum(mesh1.bounds[1], mesh2.bounds[1])
        ])
        combined_extents = combined_bounds[1] - combined_bounds[0]
        bbox_volume = np.prod(combined_extents)
        
        if bbox_volume <= 1e-9:
            print("    Error: Combined bounding box has zero volume")
            return None, None, None, time.time() - start_time, None
        
        # Generate random samples in the combined bounding box
        random_points = np.random.random((num_samples, 3))
        random_points = random_points * combined_extents + combined_bounds[0]
        
        # Process points in batches to reduce memory usage
        batch_size = 1000
        sdf_start_time = time.time()
        
        # Initialize arrays to store all SDF results
        sdf1_all = np.zeros(num_samples)
        sdf2_all = np.zeros(num_samples)
        
        # Process points in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_points = random_points[i:end_idx]
            
            # Compute signed distances for this batch
            batch_sdf1 = trimesh.proximity.signed_distance(mesh1, batch_points)
            batch_sdf2 = trimesh.proximity.signed_distance(mesh2, batch_points)
            
            sdf1_all[i:end_idx] = batch_sdf1
            sdf2_all[i:end_idx] = batch_sdf2
            
            # Optional: Print progress for large datasets
            if num_samples > 5000 and (i // batch_size) % 5 == 0:
                progress = (end_idx / num_samples) * 100
                print(f"    Batch progress: {progress:.1f}% ({end_idx}/{num_samples} points)")
        
        sdf_time = time.time() - sdf_start_time
        
        # Points inside both meshes (SDF >= 0 for both)
        inside_both = (sdf1_all >= 0) & (sdf2_all >= 0)
        num_inside_both = np.sum(inside_both)
        
        # Estimate intersection volume
        fraction_inside = num_inside_both / num_samples
        intersection_volume = fraction_inside * bbox_volume
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        computation_time = time.time() - start_time
        
        print(f"    Combined bounding box volume: {bbox_volume:.6f}")
        print(f"    Points inside both meshes: {num_inside_both}/{num_samples}")
        print(f"    Fraction inside both: {fraction_inside:.6f}")
        print(f"    Intersection volume: {intersection_volume:.6f}")
        print(f"    SDF computation time: {sdf_time:.4f}s")
        print(f"    Total computation time: {computation_time:.4f}s")
        
        return intersection_volume, num_inside_both, bbox_volume, computation_time, current_rss_mb
        
    except Exception as e:
        print(f"    Error in Monte Carlo method: {e}")
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return None, None, None, time.time() - start_time, current_rss_mb

# --- Visualization Functions ---
def visualize_intersection_matplotlib(mesh1, mesh2, test_case_name, mesh1_name, mesh2_name):
    """Visualize mesh intersection using matplotlib."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Mesh1
        ax1 = fig.add_subplot(131, projection='3d')
        vertices1 = mesh1.vertices
        ax1.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], 
                   c='blue', alpha=0.6, s=0.1, label=mesh1_name)
        ax1.set_title(f'{mesh1_name}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Mesh2
        ax2 = fig.add_subplot(132, projection='3d')
        vertices2 = mesh2.vertices
        ax2.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
                   c='red', alpha=0.6, s=0.1, label=mesh2_name)
        ax2.set_title(f'{mesh2_name}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot 3: Both meshes together
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], 
                   c='blue', alpha=0.4, s=0.1, label=mesh1_name)
        ax3.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
                   c='red', alpha=0.4, s=0.1, label=mesh2_name)
        ax3.set_title('Both Meshes')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.suptitle(f'Intersection Test Case: {test_case_name}')
        plt.tight_layout()
        
        output_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                 f"{mesh1_name}_{mesh2_name}_{test_case_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved visualization: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"    Error creating matplotlib visualization: {e}")

def visualize_intersection_open3d(mesh1, mesh2, test_case_name, mesh1_name, mesh2_name):
    """Save mesh intersection for Open3D visualization."""
    if not OPEN3D_AVAILABLE or not ENABLE_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        # Convert trimesh to open3d meshes
        o3d_mesh1 = o3d.geometry.TriangleMesh()
        o3d_mesh1.vertices = o3d.utility.Vector3dVector(mesh1.vertices)
        o3d_mesh1.triangles = o3d.utility.Vector3iVector(mesh1.faces)
        o3d_mesh1.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        o3d_mesh1.compute_vertex_normals()
        
        o3d_mesh2 = o3d.geometry.TriangleMesh()
        o3d_mesh2.vertices = o3d.utility.Vector3dVector(mesh2.vertices)
        o3d_mesh2.triangles = o3d.utility.Vector3iVector(mesh2.faces)
        o3d_mesh2.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        o3d_mesh2.compute_vertex_normals()
        
        # Save individual meshes
        output_file1 = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                   f"{mesh1_name}_{mesh2_name}_{test_case_name}_mesh1.ply")
        output_file2 = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                   f"{mesh1_name}_{mesh2_name}_{test_case_name}_mesh2.ply")
        
        o3d.io.write_triangle_mesh(output_file1, o3d_mesh1)
        o3d.io.write_triangle_mesh(output_file2, o3d_mesh2)
        
        print(f"    Saved Open3D meshes: {output_file1}, {output_file2}")
        
    except Exception as e:
        print(f"    Error creating Open3D visualization: {e}")

# --- Main Execution Logic ---
def run_intersection_tests():
    """Run intersection volume estimation tests."""
    mesh_files = get_mesh_files(MESH_DIR)
    if not mesh_files:
        print("No mesh files found!")
        return
    
    all_results = {}
    
    # Filter available mesh pairs
    available_pairs = []
    for mesh1_name, mesh2_name in TEST_MESH_PAIRS:
        if mesh1_name in mesh_files and mesh2_name in mesh_files:
            available_pairs.append((mesh1_name, mesh2_name))
        else:
            print(f"Skipping pair ({mesh1_name}, {mesh2_name}) - files not found")
    
    if not available_pairs:
        print("No valid mesh pairs found!")
        return
    
    for mesh1_name, mesh2_name in available_pairs:
        print(f"\n{'='*60}")
        print(f"Processing Mesh Pair: {mesh1_name} & {mesh2_name}")
        print(f"{'='*60}")
        
        # Load meshes
        try:
            mesh1_path = os.path.join(MESH_DIR, mesh1_name)
            mesh2_path = os.path.join(MESH_DIR, mesh2_name)
            
            print(f"Loading {mesh1_name}...")
            mesh1_original = trimesh.load_mesh(mesh1_path, process=True)
            print(f"Loading {mesh2_name}...")
            mesh2_original = trimesh.load_mesh(mesh2_path, process=True)
            
            print(f"Mesh1: {len(mesh1_original.vertices)} vertices, {len(mesh1_original.faces)} faces")
            print(f"Mesh2: {len(mesh2_original.vertices)} vertices, {len(mesh2_original.faces)} faces")
            
        except Exception as e:
            print(f"Error loading meshes: {e}")
            continue
        
        # Generate test cases
        test_cases = generate_test_cases(mesh1_original, mesh2_original, mesh1_name, mesh2_name)
        
        pair_key = f"{mesh1_name}_{mesh2_name}"
        all_results[pair_key] = {
            'test_cases': [],
            'mesh1_volume': mesh1_original.volume,
            'mesh2_volume': mesh2_original.volume
        }
        
        for test_case in test_cases:
            print(f"\n--- Test Case: {test_case['name']} ---")
            print(f"Description: {test_case['description']}")
            
            # Apply transformations
            mesh1_transform = test_case.get('mesh1_transform', {})
            mesh2_transform = test_case.get('mesh2_transform', {})
            
            mesh1 = apply_transform_to_mesh(mesh1_original, **mesh1_transform)
            mesh2 = apply_transform_to_mesh(mesh2_original, **mesh2_transform)
            
            # Create visualizations
            if ENABLE_VISUALIZATION:
                visualize_intersection_matplotlib(mesh1, mesh2, test_case['name'], 
                                                mesh1_name, mesh2_name)
                visualize_intersection_open3d(mesh1, mesh2, test_case['name'], 
                                             mesh1_name, mesh2_name)
            
            test_case_results = {
                'name': test_case['name'],
                'description': test_case['description'],
                'expected_ratio': test_case['expected_ratio'],
                'voxelization': [],
                'monte_carlo': []
            }
            
            # Test voxelization method
            print("\nVoxelization Method:")
            for resolution in VOXEL_RESOLUTIONS:
                volume, count, time_taken, memory = estimate_intersection_volume_voxelization(
                    mesh1, mesh2, resolution)
                
                if volume is not None:
                    test_case_results['voxelization'].append({
                        'resolution': resolution,
                        'volume': volume,
                        'voxel_count': count,
                        'time': time_taken,
                        'memory_mb': memory
                    })
            
            # Test Monte Carlo method
            print("\nMonte Carlo Method:")
            for num_samples in MONTE_CARLO_SAMPLES:
                volume, inside_count, bbox_vol, time_taken, memory = estimate_intersection_volume_monte_carlo(
                    mesh1, mesh2, num_samples)
                
                if volume is not None:
                    test_case_results['monte_carlo'].append({
                        'samples': num_samples,
                        'volume': volume,
                        'inside_count': inside_count,
                        'bbox_volume': bbox_vol,
                        'time': time_taken,
                        'memory_mb': memory
                    })
            
            all_results[pair_key]['test_cases'].append(test_case_results)
    
    # Generate report
    generate_intersection_report(all_results)
    
    return all_results

def generate_intersection_report(all_results, report_filename="intersection_volume_report.md"):
    """Generate a comprehensive markdown report for intersection volume estimation."""
    report_parts = []
    
    # Title and Introduction
    report_parts.append("# Mesh Intersection Volume Estimation Report\n\n")
    report_parts.append("This report presents the results of mesh intersection volume estimation using two methods:\n")
    report_parts.append("1. **Voxelization Method**: Voxelize both meshes and count overlapping voxels\n")
    report_parts.append("2. **Monte Carlo Sampling**: Sample points in bounding box and check if inside both meshes\n\n")
    
    report_parts.append("## Methodology\n\n")
    report_parts.append("### Test Case Generation\n")
    report_parts.append("For each mesh pair, we generated 6 test cases with varying intersection ratios:\n")
    report_parts.append("- **No Intersection**: Meshes separated with no overlap (target ratio: 0%)\n")
    report_parts.append("- **Small Intersection**: Small overlap (target ratio: ~15%)\n")
    report_parts.append("- **Medium Intersection**: Medium overlap (target ratio: ~40%)\n")
    report_parts.append("- **Large Intersection**: Large overlap (target ratio: ~70%)\n")
    report_parts.append("- **Maximum Intersection**: One mesh inside another (target ratio: ~90%)\n")
    report_parts.append("- **Rotated Intersection**: Meshes with rotation and medium overlap (target ratio: ~30%)\n\n")
    
    report_parts.append("### Voxelization Method\n")
    report_parts.append("1. Compute combined bounding box of both meshes\n")
    report_parts.append("2. Voxelize both meshes using the same pitch and coordinate system\n")
    report_parts.append("3. Fill voxel grids to create solid volumes\n")
    report_parts.append("4. Compute intersection as overlapping voxels\n")
    report_parts.append("5. Volume = (Number of overlapping voxels) × (Voxel size)³\n\n")
    
    report_parts.append("### Monte Carlo Method\n")
    report_parts.append("1. Compute combined bounding box of both meshes\n")
    report_parts.append("2. Randomly sample points within the bounding box\n")
    report_parts.append("3. Use Signed Distance Field (SDF) to check if points are inside both meshes\n")
    report_parts.append("4. Volume ≈ (Fraction of points inside both) × (Bounding box volume)\n\n")
    
    # Results for each mesh pair
    report_parts.append("## Results by Mesh Pair\n\n")
    
    for pair_key, pair_data in all_results.items():
        mesh1_vol = pair_data.get('mesh1_volume', 'N/A')
        mesh2_vol = pair_data.get('mesh2_volume', 'N/A')
        
        report_parts.append(f"### {pair_key.replace('_', ' & ')}\n")
        report_parts.append(f"**Mesh 1 Volume**: {mesh1_vol:.6f} | **Mesh 2 Volume**: {mesh2_vol:.6f}\n\n")
        
        for test_case in pair_data['test_cases']:
            report_parts.append(f"#### Test Case: {test_case['name']}\n")
            report_parts.append(f"**Description**: {test_case['description']}\n")
            report_parts.append(f"**Expected Intersection Ratio**: {test_case['expected_ratio']:.1%}\n\n")
            
            # Voxelization results
            if test_case['voxelization']:
                report_parts.append("**Voxelization Results:**\n\n")
                header = "| Resolution | Volume | Voxel Count | Time (s) |"
                separator = "|------------|--------|-------------|----------|"
                if PSUTIL_AVAILABLE and test_case['voxelization'][0].get('memory_mb') is not None:
                    header += " Memory (MB) |"
                    separator += "-------------|"
                report_parts.append(header + "\n")
                report_parts.append(separator + "\n")
                
                for result in test_case['voxelization']:
                    row = f"| {result['resolution']} | {result['volume']:.6f} | {result['voxel_count']} | {result['time']:.4f} |"
                    if result.get('memory_mb') is not None:
                        row += f" {result['memory_mb']:.2f} |"
                    report_parts.append(row + "\n")
                report_parts.append("\n")
            
            # Monte Carlo results
            if test_case['monte_carlo']:
                report_parts.append("**Monte Carlo Results:**\n\n")
                header = "| Samples | Volume | Inside Count | BBox Volume | Time (s) |"
                separator = "|---------|--------|--------------|-------------|----------|"
                if PSUTIL_AVAILABLE and test_case['monte_carlo'][0].get('memory_mb') is not None:
                    header += " Memory (MB) |"
                    separator += "-------------|"
                report_parts.append(header + "\n")
                report_parts.append(separator + "\n")
                
                for result in test_case['monte_carlo']:
                    row = f"| {result['samples']} | {result['volume']:.6f} | {result['inside_count']} | {result['bbox_volume']:.4f} | {result['time']:.4f} |"
                    if result.get('memory_mb') is not None:
                        row += f" {result['memory_mb']:.2f} |"
                    report_parts.append(row + "\n")
                report_parts.append("\n")
    
    # Analysis section
    report_parts.append("## Analysis and Discussion\n\n")
    report_parts.append("### Accuracy Analysis\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- **Voxelization Accuracy**: How does accuracy change with resolution?\n")
    report_parts.append("- **Monte Carlo Accuracy**: How does accuracy improve with sample count?\n")
    report_parts.append("- **Method Comparison**: Which method provides better accuracy for different intersection scenarios?\n")
    report_parts.append("- **Edge Cases**: How do methods perform with no intersection vs. maximum intersection?\n\n")
    
    report_parts.append("### Computational Cost Analysis\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- **Voxelization Performance**: How does computation time scale with resolution?\n")
    report_parts.append("- **Monte Carlo Performance**: How does computation time scale with sample count?\n")
    report_parts.append("- **Memory Usage**: Which method is more memory efficient?\n")
    report_parts.append("- **Speed vs. Accuracy Trade-offs**: Optimal parameters for different use cases\n\n")
    
    report_parts.append("### Method Comparison Summary\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("| Aspect | Voxelization | Monte Carlo |\n")
    report_parts.append("|--------|--------------|-------------|\n")
    report_parts.append("| **Accuracy** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Speed** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Memory** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Scalability** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Best Use Case** | [Analysis needed] | [Analysis needed] |\n\n")
    
    report_parts.append("### Conclusions\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- Key findings from the intersection volume estimation experiments\n")
    report_parts.append("- Recommendations for method selection based on requirements\n")
    report_parts.append("- Limitations and potential improvements\n")
    
    # Write report
    try:
        output_path = os.path.join(os.path.dirname(__file__) or '.', report_filename)
        with open(output_path, "w") as f:
            f.write("".join(report_parts))
        print(f"\nIntersection volume estimation report generated: {os.path.abspath(output_path)}")
    except IOError as e:
        print(f"Error writing report to {report_filename}: {e}")

def main():
    """Main function to run intersection volume estimation tests."""
    print("Starting Mesh Intersection Volume Estimation Tests")
    print("=" * 60)
    
    # Check if mesh directory exists
    if not os.path.exists(MESH_DIR):
        print(f"Error: Mesh directory '{MESH_DIR}' not found.")
        print("Please ensure the mesh directory is correctly configured.")
        return
    
    # Run tests
    results = run_intersection_tests()
    
    if results:
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print(f"Results saved to: {VISUALIZATION_OUTPUT_DIR}")
        print("Check the generated markdown report for detailed analysis.")
    else:
        print("No tests were completed successfully.")

if __name__ == "__main__":
    main()
