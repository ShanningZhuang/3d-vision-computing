import trimesh
import numpy as np
import time
import os
import math
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import itertools

# Add visualization imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("matplotlib library found. Will generate sphere approximation plots.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib library not found. Sphere approximation plots will not be generated.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("open3d library found. Will generate 3D visualizations.")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("open3d library not found. 3D visualizations will not be generated.")

# Attempt to import psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil library found. Will attempt to record memory usage.")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil library not found. Memory usage will not be recorded.")

# --- Configuration ---
MESH_DIR = "../../objs_approx"

# Sphere approximation parameters
MAX_SPHERE_COUNTS = [200,500,1000]  # Different values of N
SPHERE_RADIUS_RATIOS = [0.05, 0.1, 0.15, 0.2]  # Radius as fraction of mesh's max extent

# Visualization settings
ENABLE_VISUALIZATION = True
VISUALIZATION_OUTPUT_DIR = "./sphere_approximation_debug"

# Optimization parameters
OPTIMIZATION_MAX_ITER = 1000
OPTIMIZATION_TOLERANCE = 1e-6

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

# --- Sphere Utility Functions ---
def sphere_volume(radius):
    """Calculate volume of a sphere."""
    return (4.0 / 3.0) * np.pi * (radius ** 3)

def sphere_intersection_volume(center1, center2, radius):
    """Calculate intersection volume of two spheres with same radius."""
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    
    # No intersection
    if distance >= 2 * radius:
        return 0.0
    
    # One sphere completely inside another (same radius means complete overlap)
    if distance <= 0:
        return sphere_volume(radius)
    
    # Partial intersection - use spherical cap formula
    # Volume of intersection of two equal spheres
    h = radius - distance / 2.0  # Height of spherical cap
    if h <= 0:
        return 0.0
    
    # Volume of spherical cap: V = π * h² * (3r - h) / 3
    cap_volume = np.pi * (h ** 2) * (3 * radius - h) / 3.0
    intersection_volume = 2 * cap_volume
    
    return min(intersection_volume, sphere_volume(radius))

def calculate_union_volume_inclusion_exclusion(sphere_centers, radius, max_terms=None):
    """
    Calculate union volume using inclusion-exclusion principle.
    For large numbers of spheres, we can limit the number of terms to consider.
    """
    n_spheres = len(sphere_centers)
    if n_spheres == 0:
        return 0.0
    
    if n_spheres == 1:
        return sphere_volume(radius)
    
    # Limit computation for large numbers of spheres
    if max_terms is None:
        max_terms = min(n_spheres, 10)  # Limit to avoid exponential explosion
    
    total_volume = 0.0
    
    # Inclusion-exclusion principle: sum over all non-empty subsets
    for k in range(1, min(max_terms + 1, n_spheres + 1)):
        sign = (-1) ** (k - 1)
        
        # Sum over all k-combinations
        subset_volume = 0.0
        count = 0
        max_combinations = 1000  # Limit combinations to avoid excessive computation
        
        for subset in itertools.combinations(range(n_spheres), k):
            if count >= max_combinations:
                # Estimate remaining combinations
                remaining_combinations = math.comb(n_spheres, k) - count
                if remaining_combinations > 0:
                    avg_intersection = subset_volume / count if count > 0 else 0
                    subset_volume += avg_intersection * remaining_combinations
                break
            
            # Calculate intersection volume of this subset
            intersection_vol = calculate_k_sphere_intersection(
                [sphere_centers[i] for i in subset], radius)
            subset_volume += intersection_vol
            count += 1
        
        total_volume += sign * subset_volume
    
    return max(0.0, total_volume)

def calculate_k_sphere_intersection(centers, radius):
    """Calculate intersection volume of k spheres (approximation for k > 2)."""
    if len(centers) == 1:
        return sphere_volume(radius)
    elif len(centers) == 2:
        return sphere_intersection_volume(centers[0], centers[1], radius)
    else:
        # For k > 2, use approximation based on pairwise distances
        # This is a simplified approximation - exact calculation is complex
        centers = np.array(centers)
        distances = cdist(centers, centers)
        
        # If any pair is too far apart, no intersection
        max_distance = 2 * radius
        if np.any(distances > max_distance):
            return 0.0
        
        # Rough approximation: scale down based on number of spheres and distances
        base_volume = sphere_volume(radius)
        avg_distance = np.mean(distances[distances > 0])
        overlap_factor = max(0, 1 - avg_distance / (2 * radius))
        
        # Scale down exponentially with number of spheres
        k_factor = overlap_factor ** (len(centers) - 1)
        return base_volume * k_factor

def is_sphere_inside_mesh(center, radius, mesh):
    """Check if a sphere is entirely inside the mesh using signed distance field."""
    try:
        # Calculate signed distance from sphere center to mesh boundary
        sdf = trimesh.proximity.signed_distance(mesh, [center])[0]
        
        # If SDF >= radius, then the entire sphere lies within the mesh
        # (the closest point on mesh boundary is at least 'radius' distance away)
        return sdf >= radius
    except:
        return False

def generate_candidate_points(mesh, num_candidates=1000):
    """Generate candidate points inside the mesh for sphere placement using batch processing."""
    min_bound, max_bound = mesh.bounds
    bbox_size = max_bound - min_bound
    
    candidates = []
    batch_size = min(10000, num_candidates * 5)  # Generate in batches
    max_attempts = 10  # Limit total attempts to avoid infinite loops
    
    for attempt in range(max_attempts):
        if len(candidates) >= num_candidates:
            break
            
        # Generate a batch of random points in bounding box
        remaining_needed = num_candidates - len(candidates)
        current_batch_size = min(batch_size, remaining_needed * 3)  # Generate extra to account for filtering
        
        # Vectorized random point generation
        random_points = np.random.uniform(
            low=min_bound, 
            high=max_bound, 
            size=(current_batch_size, 3)
        )
        
        try:
            # Batch SDF calculation
            sdf_values = trimesh.proximity.signed_distance(mesh, random_points)
            
            # Filter points that are inside the mesh (SDF >= 0)
            inside_mask = sdf_values >= 0
            valid_points = random_points[inside_mask]
            
            # Add valid points to candidates
            if len(valid_points) > 0:
                candidates.extend(valid_points)
                
        except Exception as e:
            print(f"    Warning: SDF calculation failed in batch {attempt + 1}: {e}")
            # Fallback: try individual point checking for this batch
            for point in random_points[:min(100, len(random_points))]:  # Limit fallback
                try:
                    sdf = trimesh.proximity.signed_distance(mesh, [point])[0]
                    if sdf >= 0:
                        candidates.append(point)
                        if len(candidates) >= num_candidates:
                            break
                except:
                    continue
    
    # Convert to numpy array and limit to requested number
    candidates = np.array(candidates)
    if len(candidates) > num_candidates:
        # Randomly sample to get exactly num_candidates points
        indices = np.random.choice(len(candidates), num_candidates, replace=False)
        candidates = candidates[indices]
    
    return candidates

# --- Greedy Algorithm ---
def greedy_sphere_placement(mesh, radius, max_spheres, mesh_name=""):
    """
    Greedy algorithm for sphere placement.
    Iteratively place spheres in the largest uncovered regions.
    """
    print(f"  Greedy Algorithm: radius={radius:.4f}, max_spheres={max_spheres}")
    
    start_time = time.time()
    current_rss_mb = None
    
    try:
        # Generate candidate points inside the mesh
        print(f"    Generating candidate points...")
        candidates = generate_candidate_points(mesh, num_candidates=5000)  # Increased for better coverage
        
        if len(candidates) == 0:
            print(f"    Error: No valid candidate points found inside mesh")
            return [], 0.0, time.time() - start_time, current_rss_mb
        
        print(f"    Found {len(candidates)} candidate points")
        
        # Pre-filter candidates: check which ones can form valid spheres (batch processing)
        print(f"    Pre-filtering valid sphere positions...")
        candidate_sdf = trimesh.proximity.signed_distance(mesh, candidates)
        valid_mask = candidate_sdf >= radius
        valid_candidates = candidates[valid_mask]
        
        if len(valid_candidates) == 0:
            print(f"    Error: No valid sphere positions found")
            return [], 0.0, time.time() - start_time, current_rss_mb
        
        print(f"    Found {len(valid_candidates)} valid sphere positions")
        
        placed_spheres = []
        
        for sphere_idx in range(max_spheres):
            print(f"    Placing sphere {sphere_idx + 1}/{max_spheres}...")
            
            if len(valid_candidates) == 0:
                print(f"      No more valid candidates available")
                break
            
            if len(placed_spheres) == 0:
                # First sphere: just pick the first valid candidate
                best_idx = 0
                best_center = valid_candidates[best_idx].copy()
                best_score = 1.0
                found_non_overlapping = True
                print(f"      Placed first sphere at {best_center}")
            else:
                # Calculate distances from all valid candidates to all placed spheres (vectorized)
                placed_spheres_array = np.array(placed_spheres)  # Shape: (n_placed, 3)
                
                # Calculate pairwise distances: candidates vs placed spheres
                # valid_candidates shape: (n_candidates, 3)
                # placed_spheres_array shape: (n_placed, 3)
                # distances shape: (n_candidates, n_placed)
                distances = cdist(valid_candidates, placed_spheres_array)
                
                # Find minimum distance to any placed sphere for each candidate
                min_distances = np.min(distances, axis=1)  # Shape: (n_candidates,)
                
                # Check for non-overlapping spheres (distance >= 2*radius)
                non_overlapping_mask = min_distances >= 2 * radius
                
                if np.any(non_overlapping_mask):
                    # Found non-overlapping positions - pick the first one
                    non_overlapping_indices = np.where(non_overlapping_mask)[0]
                    best_idx = non_overlapping_indices[0]
                    best_center = valid_candidates[best_idx].copy()
                    best_score = float('inf')
                    found_non_overlapping = True
                    print(f"      Found non-overlapping position at {best_center} (min_distance: {min_distances[best_idx]:.4f})")
                else:
                    # No non-overlapping positions - find best score (vectorized)
                    scores = min_distances / (2 * radius)  # Normalize by sphere diameter
                    best_idx = np.argmax(scores)
                    best_center = valid_candidates[best_idx].copy()
                    best_score = scores[best_idx]
                    found_non_overlapping = False
                    print(f"      Placed sphere at {best_center} with score {best_score:.4f}")
            
            # Add the selected sphere
            placed_spheres.append(best_center)
            
            # Remove the selected candidate from valid_candidates to avoid reselection
            valid_candidates = np.delete(valid_candidates, best_idx, axis=0)
            
            # Optional: Remove candidates that are too close to the newly placed sphere
            # This can help avoid clustering and improve coverage
            if len(valid_candidates) > 0:
                distances_to_new = np.linalg.norm(valid_candidates - best_center, axis=1)
                # Keep candidates that are at least radius distance away (to avoid tight clustering)
                keep_mask = distances_to_new >= radius
                valid_candidates = valid_candidates[keep_mask]
                
                if len(valid_candidates) == 0:
                    print(f"      No more valid candidates after filtering near new sphere")
        
        # Calculate coverage
        if len(placed_spheres) > 0:
            union_volume = calculate_union_volume_inclusion_exclusion(placed_spheres, radius)
            coverage_percentage = (union_volume / mesh.volume) * 100 if mesh.volume > 0 else 0
        else:
            union_volume = 0.0
            coverage_percentage = 0.0
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        computation_time = time.time() - start_time
        
        print(f"    Placed {len(placed_spheres)} spheres")
        print(f"    Union volume: {union_volume:.6f}")
        print(f"    Coverage: {coverage_percentage:.2f}%")
        print(f"    Computation time: {computation_time:.4f}s")
        
        return placed_spheres, coverage_percentage, computation_time, current_rss_mb
        
    except Exception as e:
        print(f"    Error in greedy algorithm: {e}")
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return [], 0.0, time.time() - start_time, current_rss_mb

# --- Optimization Algorithm ---
def optimization_sphere_placement(mesh, radius, max_spheres, mesh_name=""):
    """
    Optimization-based sphere placement using hybrid approach.
    For large sphere counts, use iterative local optimization instead of global optimization.
    """
    print(f"  Optimization Algorithm: radius={radius:.4f}, max_spheres={max_spheres}")
    
    start_time = time.time()
    current_rss_mb = None
    
    try:
        # Generate initial candidate points using our optimized batch method
        candidates = generate_candidate_points(mesh, num_candidates=min(5000, max_spheres * 10))
        
        if len(candidates) == 0:
            print(f"    Error: No valid candidate points found inside mesh")
            return [], 0.0, time.time() - start_time, current_rss_mb
        
        # For large sphere counts, use a hybrid greedy-optimization approach
        if max_spheres > 50:
            print(f"    Using hybrid approach for large sphere count ({max_spheres})")
            placed_spheres = hybrid_sphere_placement(mesh, radius, max_spheres, candidates)
        else:
            print(f"    Using differential evolution for small sphere count ({max_spheres})")
            placed_spheres = differential_evolution_placement(mesh, radius, max_spheres, candidates)
        
        # Calculate final coverage
        if len(placed_spheres) > 0:
            union_volume = calculate_union_volume_inclusion_exclusion(placed_spheres, radius, max_terms=5)
            coverage_percentage = (union_volume / mesh.volume) * 100 if mesh.volume > 0 else 0
        else:
            union_volume = 0.0
            coverage_percentage = 0.0
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        computation_time = time.time() - start_time
        
        print(f"    Placed {len(placed_spheres)} spheres")
        print(f"    Union volume: {union_volume:.6f}")
        print(f"    Coverage: {coverage_percentage:.2f}%")
        print(f"    Computation time: {computation_time:.4f}s")
        
        return placed_spheres, coverage_percentage, computation_time, current_rss_mb
        
    except Exception as e:
        print(f"    Error in optimization algorithm: {e}")
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            current_rss_mb = process.memory_info().rss / (1024 * 1024)
        return [], 0.0, time.time() - start_time, current_rss_mb

def hybrid_sphere_placement(mesh, radius, max_spheres, candidates):
    """
    Hybrid approach: Start with greedy placement, then optimize in batches.
    Much more efficient for large sphere counts.
    """
    print(f"      Starting with greedy initialization...")
    
    # Pre-filter candidates
    candidate_sdf = trimesh.proximity.signed_distance(mesh, candidates)
    valid_mask = candidate_sdf >= radius
    valid_candidates = candidates[valid_mask]
    
    if len(valid_candidates) == 0:
        return []
    
    placed_spheres = []
    
    # Phase 1: Greedy placement for initial coverage
    initial_count = min(max_spheres, len(valid_candidates))
    
    for i in range(initial_count):
        if len(valid_candidates) == 0:
            break
            
        if len(placed_spheres) == 0:
            # First sphere
            best_idx = 0
        else:
            # Find sphere that maximizes minimum distance to existing spheres
            placed_array = np.array(placed_spheres)
            distances = cdist(valid_candidates, placed_array)
            min_distances = np.min(distances, axis=1)
            
            # Prefer non-overlapping positions
            non_overlapping = min_distances >= 2 * radius
            if np.any(non_overlapping):
                non_overlapping_indices = np.where(non_overlapping)[0]
                best_idx = non_overlapping_indices[0]
            else:
                best_idx = np.argmax(min_distances)
        
        # Add sphere and remove from candidates
        placed_spheres.append(valid_candidates[best_idx])
        valid_candidates = np.delete(valid_candidates, best_idx, axis=0)
        
        # Progress reporting for large counts
        if i % 100 == 0 and i > 0:
            print(f"        Placed {i}/{initial_count} spheres...")
    
    print(f"      Greedy phase completed: {len(placed_spheres)} spheres placed")
    
    # Phase 2: Local optimization in batches (only if we have reasonable number of spheres)
    if len(placed_spheres) > 10 and len(placed_spheres) <= 200:
        print(f"      Starting local optimization phase...")
        placed_spheres = local_batch_optimization(mesh, radius, placed_spheres)
        print(f"      Local optimization completed: {len(placed_spheres)} spheres")
    
    return placed_spheres

def local_batch_optimization(mesh, radius, initial_spheres, batch_size=20):
    """
    Optimize sphere positions in small batches using local optimization.
    """
    spheres = [sphere.copy() for sphere in initial_spheres]
    min_bound, max_bound = mesh.bounds
    
    # Optimize in batches
    num_batches = (len(spheres) + batch_size - 1) // batch_size
    
    for batch_idx in range(min(num_batches, 5)):  # Limit to 5 batches for time
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(spheres))
        
        if start_idx >= len(spheres):
            break
            
        print(f"        Optimizing batch {batch_idx + 1}/{min(num_batches, 5)} (spheres {start_idx}-{end_idx-1})")
        
        # Extract batch
        batch_spheres = spheres[start_idx:end_idx]
        other_spheres = spheres[:start_idx] + spheres[end_idx:]
        
        # Define local objective for this batch
        def batch_objective(x):
            batch_centers = x.reshape(-1, 3)
            
            # Check bounds
            if np.any(batch_centers < min_bound + radius) or np.any(batch_centers > max_bound - radius):
                return 1e6
            
            # Check if spheres are inside mesh
            try:
                sdf_values = trimesh.proximity.signed_distance(mesh, batch_centers)
                if np.any(sdf_values < radius):
                    return 1e6
            except:
                return 1e6
            
            # Calculate volume contribution of this batch
            all_centers = list(other_spheres) + list(batch_centers)
            volume = calculate_union_volume_inclusion_exclusion(all_centers, radius, max_terms=3)
            return -volume
        
        # Set up bounds for batch
        bounds = []
        for _ in range(len(batch_spheres)):
            for dim in range(3):
                bounds.append((min_bound[dim] + radius, max_bound[dim] - radius))
        
        # Initial guess
        x0 = np.array(batch_spheres).flatten()
        
        # Quick local optimization
        try:
            from scipy.optimize import minimize
            result = minimize(
                batch_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )
            
            if result.success:
                optimized_batch = result.x.reshape(-1, 3)
                # Update spheres with optimized positions
                for i, new_pos in enumerate(optimized_batch):
                    spheres[start_idx + i] = new_pos
        except:
            # If optimization fails, keep original positions
            pass
    
    return spheres

def differential_evolution_placement(mesh, radius, max_spheres, candidates):
    """
    Original differential evolution approach for small sphere counts.
    """
    min_bound, max_bound = mesh.bounds
    
    # Smart initial guess
    if len(candidates) >= max_spheres:
        initial_centers = []
        remaining_candidates = candidates.copy()
        
        # First point
        first_idx = np.random.randint(len(remaining_candidates))
        initial_centers.append(remaining_candidates[first_idx])
        remaining_candidates = np.delete(remaining_candidates, first_idx, axis=0)
        
        # Subsequent points: maximize minimum distance
        for _ in range(max_spheres - 1):
            if len(remaining_candidates) == 0:
                break
            placed_centers = np.array(initial_centers)
            distances = cdist(remaining_candidates, placed_centers)
            min_distances = np.min(distances, axis=1)
            best_idx = np.argmax(min_distances)
            initial_centers.append(remaining_candidates[best_idx])
            remaining_candidates = np.delete(remaining_candidates, best_idx, axis=0)
        
        initial_centers = np.array(initial_centers)
    else:
        initial_centers = candidates[:max_spheres] if len(candidates) >= max_spheres else \
                         np.vstack([candidates, candidates[:max_spheres - len(candidates)]])
    
    # Define objective function
    def objective_function(x):
        centers = x.reshape(max_spheres, 3)
        if np.any(centers < min_bound + radius) or np.any(centers > max_bound - radius):
            return 1e6
        
        try:
            sdf_values = trimesh.proximity.signed_distance(mesh, centers)
            if np.any(sdf_values < radius):
                return 1e6
        except:
            return 1e6
        
        union_volume = calculate_union_volume_inclusion_exclusion(centers, radius, max_terms=3)
        return -union_volume
    
    # Set up bounds
    bounds = []
    for _ in range(max_spheres):
        for dim in range(3):
            bounds.append((min_bound[dim] + radius, max_bound[dim] - radius))
    
    x0 = initial_centers.flatten()
    
    # Run differential evolution with reduced parameters
    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=50,  # Reduced iterations
        popsize=10,  # Reduced population
        seed=42,
        atol=1e-4,
        tol=1e-4
    )
    
    if result.success:
        optimal_centers = result.x.reshape(max_spheres, 3)
        # Validate results
        try:
            center_sdf = trimesh.proximity.signed_distance(mesh, optimal_centers)
            valid_mask = center_sdf >= radius
            valid_spheres = optimal_centers[valid_mask]
            return valid_spheres.tolist()
        except:
            return []
    
    return []

# --- Visualization Functions ---
def visualize_sphere_approximation_matplotlib(mesh, spheres, radius, method_name, mesh_name, 
                                            max_spheres, coverage_percentage):
    """Visualize sphere approximation using matplotlib."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Mesh wireframe
        ax1 = fig.add_subplot(131, projection='3d')
        vertices = mesh.vertices
        ax1.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2], 
                   c='lightblue', alpha=0.3, s=0.1, label='Mesh vertices')
        ax1.set_title(f'Original Mesh: {mesh_name}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Spheres only
        ax2 = fig.add_subplot(132, projection='3d')
        if len(spheres) > 0:
            spheres_array = np.array(spheres)
            ax2.scatter(spheres_array[:, 0], spheres_array[:, 1], spheres_array[:, 2], 
                       c='red', s=100, alpha=0.8, label=f'{len(spheres)} spheres')
            
            # Draw sphere outlines
            for i, center in enumerate(spheres):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax2.plot_surface(x, y, z, alpha=0.2, color='red')
        
        ax2.set_title(f'Sphere Approximation\n{method_name}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # Plot 3: Combined view
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(vertices[::20, 0], vertices[::20, 1], vertices[::20, 2], 
                   c='lightblue', alpha=0.2, s=0.1, label='Mesh')
        
        if len(spheres) > 0:
            spheres_array = np.array(spheres)
            ax3.scatter(spheres_array[:, 0], spheres_array[:, 1], spheres_array[:, 2], 
                       c='red', s=50, alpha=0.9, label=f'{len(spheres)} spheres')
        
        ax3.set_title(f'Combined View\nCoverage: {coverage_percentage:.1f}%')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.suptitle(f'Sphere Approximation: {mesh_name} ({method_name})\n'
                    f'Max Spheres: {max_spheres}, Radius: {radius:.4f}')
        plt.tight_layout()
        
        output_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                 f"{mesh_name}_{method_name}_spheres_{max_spheres}_r{radius:.3f}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved visualization: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"    Error creating matplotlib visualization: {e}")

def visualize_sphere_approximation_open3d(mesh, spheres, radius, method_name, mesh_name, 
                                        max_spheres, coverage_percentage):
    """Save sphere approximation for Open3D visualization."""
    if not OPEN3D_AVAILABLE or not ENABLE_VISUALIZATION:
        return
    
    try:
        create_output_directory()
        
        # Convert trimesh to open3d mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
        o3d_mesh.compute_vertex_normals()
        
        # Create combined sphere geometry
        if len(spheres) > 0:
            combined_spheres = o3d.geometry.TriangleMesh()
            
            for i, center in enumerate(spheres):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                
                # Merge with combined spheres
                combined_spheres += sphere
            
            combined_spheres.compute_vertex_normals()
        
        # Save mesh
        mesh_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                f"{mesh_name}_{method_name}_mesh_{max_spheres}_r{radius:.3f}.ply")
        o3d.io.write_triangle_mesh(mesh_file, o3d_mesh)
        
        # Save combined spheres (only if there are spheres)
        if len(spheres) > 0:
            spheres_file = os.path.join(VISUALIZATION_OUTPUT_DIR, 
                                       f"{mesh_name}_{method_name}_spheres_{max_spheres}_r{radius:.3f}.ply")
            o3d.io.write_triangle_mesh(spheres_file, combined_spheres)
            print(f"    Saved Open3D files: {mesh_file} and {spheres_file}")
        else:
            print(f"    Saved Open3D mesh: {mesh_file}")
        
    except Exception as e:
        print(f"    Error creating Open3D visualization: {e}")

# --- Main Execution Logic ---
def run_sphere_approximation_tests():
    """Run sphere approximation tests on all meshes."""
    mesh_files = get_mesh_files(MESH_DIR)
    if not mesh_files:
        print("No mesh files found!")
        return
    
    all_results = {}
    
    for mesh_file in mesh_files:
        print(f"\n{'='*60}")
        print(f"Processing Mesh: {mesh_file}")
        print(f"{'='*60}")
        
        # Load mesh
        try:
            mesh_path = os.path.join(MESH_DIR, mesh_file)
            mesh = trimesh.load_mesh(mesh_path, process=True)
            
            print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            print(f"Mesh volume: {mesh.volume:.6f}")
            print(f"Mesh bounds: {mesh.bounds}")
            print(f"Mesh extents: {mesh.extents}")
            print(f"Watertight: {mesh.is_watertight}")
            
            if mesh.is_empty or mesh.volume <= 0:
                print(f"Mesh {mesh_file} is empty or has zero volume. Skipping.")
                continue
                
        except Exception as e:
            print(f"Error loading mesh {mesh_file}: {e}")
            continue
        
        mesh_name = os.path.splitext(mesh_file)[0]
        all_results[mesh_name] = {
            'mesh_volume': mesh.volume,
            'mesh_extents': mesh.extents.tolist(),
            'greedy_results': [],
            'optimization_results': []
        }
        
        # Test different sphere configurations
        for radius_ratio in SPHERE_RADIUS_RATIOS:
            radius = mesh.extents.max() * radius_ratio
            print(f"\n--- Testing with radius ratio {radius_ratio} (radius: {radius:.4f}) ---")
            
            for max_spheres in MAX_SPHERE_COUNTS:
                print(f"\nMax spheres: {max_spheres}")
                
                # Test Greedy Algorithm
                print("Greedy Algorithm:")
                greedy_spheres, greedy_coverage, greedy_time, greedy_memory = \
                    greedy_sphere_placement(mesh, radius, max_spheres, mesh_name)
                
                if ENABLE_VISUALIZATION and len(greedy_spheres) > 0:
                    visualize_sphere_approximation_matplotlib(
                        mesh, greedy_spheres, radius, "Greedy", mesh_name, 
                        max_spheres, greedy_coverage)
                    visualize_sphere_approximation_open3d(
                        mesh, greedy_spheres, radius, "Greedy", mesh_name, 
                        max_spheres, greedy_coverage)
                
                all_results[mesh_name]['greedy_results'].append({
                    'radius_ratio': radius_ratio,
                    'radius': radius,
                    'max_spheres': max_spheres,
                    'placed_spheres': len(greedy_spheres),
                    'sphere_centers': greedy_spheres,
                    'coverage_percentage': greedy_coverage,
                    'computation_time': greedy_time,
                    'memory_mb': greedy_memory
                })
                
                # Test Optimization Algorithm
                print("Optimization Algorithm:")
                opt_spheres, opt_coverage, opt_time, opt_memory = \
                    optimization_sphere_placement(mesh, radius, max_spheres, mesh_name)
                
                if ENABLE_VISUALIZATION and len(opt_spheres) > 0:
                    visualize_sphere_approximation_matplotlib(
                        mesh, opt_spheres, radius, "Optimization", mesh_name, 
                        max_spheres, opt_coverage)
                    visualize_sphere_approximation_open3d(
                        mesh, opt_spheres, radius, "Optimization", mesh_name, 
                        max_spheres, opt_coverage)
                
                all_results[mesh_name]['optimization_results'].append({
                    'radius_ratio': radius_ratio,
                    'radius': radius,
                    'max_spheres': max_spheres,
                    'placed_spheres': len(opt_spheres),
                    'sphere_centers': opt_spheres,
                    'coverage_percentage': opt_coverage,
                    'computation_time': opt_time,
                    'memory_mb': opt_memory
                })
    
    # Generate report
    generate_sphere_approximation_report(all_results)
    
    return all_results

def generate_sphere_approximation_report(all_results, report_filename="sphere_approximation_report.md"):
    """Generate comprehensive markdown report for sphere approximation results."""
    report_parts = []
    
    # Title and Introduction
    report_parts.append("# Sphere Approximation Volume Estimation Report\n\n")
    report_parts.append("This report presents results of mesh volume approximation using spheres with two approaches:\n")
    report_parts.append("1. **Greedy Algorithm**: Iteratively place spheres in largest uncovered regions\n")
    report_parts.append("2. **Optimization Algorithm**: Formulate as constrained optimization problem\n\n")
    
    report_parts.append("## Methodology\n\n")
    report_parts.append("### Problem Constraints\n")
    report_parts.append("- All spheres must lie entirely within the mesh\n")
    report_parts.append("- Number of spheres ≤ predefined maximum N\n")
    report_parts.append("- All spheres have the same radius for each test case\n\n")
    
    report_parts.append("### Coverage Calculation\n")
    report_parts.append("- **Union Volume**: Calculated using inclusion-exclusion principle\n")
    report_parts.append("- **Coverage Percentage**: (Union Volume / Mesh Volume) × 100%\n")
    report_parts.append("- **Sphere Volume**: V = (4/3)πr³\n")
    report_parts.append("- **Intersection Handling**: Pairwise intersections calculated analytically\n\n")
    
    report_parts.append("### Algorithm Details\n")
    report_parts.append("#### Greedy Algorithm\n")
    report_parts.append("1. Generate candidate points inside mesh using SDF sampling\n")
    report_parts.append("2. For each sphere position, select location maximizing distance to existing spheres\n")
    report_parts.append("3. Verify sphere lies entirely within mesh using surface sampling\n")
    report_parts.append("4. Repeat until maximum number of spheres reached\n\n")
    
    report_parts.append("#### Optimization Algorithm\n")
    report_parts.append("1. Formulate as constrained optimization: maximize union volume\n")
    report_parts.append("2. Constraints: sphere centers must be ≥ radius distance from mesh boundary\n")
    report_parts.append("3. Use differential evolution for global optimization\n")
    report_parts.append("4. Post-process to filter invalid sphere placements\n\n")
    
    # Results for each mesh
    report_parts.append("## Results by Mesh\n\n")
    
    for mesh_name, mesh_data in all_results.items():
        mesh_volume = mesh_data.get('mesh_volume', 'N/A')
        mesh_extents = mesh_data.get('mesh_extents', [0, 0, 0])
        
        report_parts.append(f"### {mesh_name}\n")
        report_parts.append(f"**Mesh Volume**: {mesh_volume:.6f}\n")
        report_parts.append(f"**Mesh Extents**: [{mesh_extents[0]:.3f}, {mesh_extents[1]:.3f}, {mesh_extents[2]:.3f}]\n\n")
        
        # Greedy Results
        if mesh_data['greedy_results']:
            report_parts.append("#### Greedy Algorithm Results\n\n")
            header = "| Radius Ratio | Max Spheres | Placed | Coverage (%) | Time (s) |"
            separator = "|--------------|-------------|--------|--------------|----------|"
            if PSUTIL_AVAILABLE and mesh_data['greedy_results'][0].get('memory_mb') is not None:
                header += " Memory (MB) |"
                separator += "-------------|"
            report_parts.append(header + "\n")
            report_parts.append(separator + "\n")
            
            for result in mesh_data['greedy_results']:
                row = f"| {result['radius_ratio']:.2f} | {result['max_spheres']} | {result['placed_spheres']} | {result['coverage_percentage']:.2f} | {result['computation_time']:.4f} |"
                if result.get('memory_mb') is not None:
                    row += f" {result['memory_mb']:.2f} |"
                report_parts.append(row + "\n")
            report_parts.append("\n")
        
        # Optimization Results
        if mesh_data['optimization_results']:
            report_parts.append("#### Optimization Algorithm Results\n\n")
            header = "| Radius Ratio | Max Spheres | Placed | Coverage (%) | Time (s) |"
            separator = "|--------------|-------------|--------|--------------|----------|"
            if PSUTIL_AVAILABLE and mesh_data['optimization_results'][0].get('memory_mb') is not None:
                header += " Memory (MB) |"
                separator += "-------------|"
            report_parts.append(header + "\n")
            report_parts.append(separator + "\n")
            
            for result in mesh_data['optimization_results']:
                row = f"| {result['radius_ratio']:.2f} | {result['max_spheres']} | {result['placed_spheres']} | {result['coverage_percentage']:.2f} | {result['computation_time']:.4f} |"
                if result.get('memory_mb') is not None:
                    row += f" {result['memory_mb']:.2f} |"
                report_parts.append(row + "\n")
            report_parts.append("\n")
        
        # Comparison table
        report_parts.append("#### Method Comparison\n")
        report_parts.append("*(Best results for each configuration)*\n\n")
        report_parts.append("| Config | Greedy Coverage | Opt Coverage | Greedy Time | Opt Time | Winner |\n")
        report_parts.append("|--------|-----------------|--------------|-------------|----------|--------|\n")
        
        # Group results by configuration
        configs = {}
        for result in mesh_data['greedy_results']:
            key = (result['radius_ratio'], result['max_spheres'])
            if key not in configs:
                configs[key] = {'greedy': None, 'opt': None}
            configs[key]['greedy'] = result
        
        for result in mesh_data['optimization_results']:
            key = (result['radius_ratio'], result['max_spheres'])
            if key not in configs:
                configs[key] = {'greedy': None, 'opt': None}
            configs[key]['opt'] = result
        
        for (radius_ratio, max_spheres), methods in configs.items():
            greedy = methods['greedy']
            opt = methods['opt']
            
            if greedy and opt:
                greedy_cov = greedy['coverage_percentage']
                opt_cov = opt['coverage_percentage']
                greedy_time = greedy['computation_time']
                opt_time = opt['computation_time']
                
                winner = "Greedy" if greedy_cov > opt_cov else "Optimization" if opt_cov > greedy_cov else "Tie"
                
                report_parts.append(f"| r={radius_ratio:.2f}, N={max_spheres} | {greedy_cov:.2f}% | {opt_cov:.2f}% | {greedy_time:.3f}s | {opt_time:.3f}s | {winner} |\n")
        
        report_parts.append("\n")
    
    # Analysis section
    report_parts.append("## Analysis and Discussion\n\n")
    report_parts.append("### Coverage Analysis\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- **Effect of Sphere Radius**: How does coverage change with sphere size?\n")
    report_parts.append("- **Effect of Sphere Count**: Diminishing returns with more spheres?\n")
    report_parts.append("- **Algorithm Comparison**: Which method achieves better coverage?\n")
    report_parts.append("- **Mesh Complexity Impact**: How do different mesh shapes affect results?\n\n")
    
    report_parts.append("### Computational Efficiency\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- **Greedy Algorithm**: Linear scaling with number of spheres\n")
    report_parts.append("- **Optimization Algorithm**: Higher computational cost but potentially better results\n")
    report_parts.append("- **Memory Usage**: Resource requirements for different configurations\n")
    report_parts.append("- **Scalability**: Performance with larger meshes and more spheres\n\n")
    
    report_parts.append("### Trade-offs and Recommendations\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("| Aspect | Greedy Algorithm | Optimization Algorithm |\n")
    report_parts.append("|--------|------------------|------------------------|\n")
    report_parts.append("| **Speed** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Coverage Quality** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Consistency** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Memory Usage** | [Analysis needed] | [Analysis needed] |\n")
    report_parts.append("| **Best Use Case** | [Analysis needed] | [Analysis needed] |\n\n")
    
    report_parts.append("### Conclusions\n")
    report_parts.append("*(To be completed based on experimental results)*\n\n")
    report_parts.append("- Key findings from sphere approximation experiments\n")
    report_parts.append("- Optimal configurations for different requirements\n")
    report_parts.append("- Limitations and potential improvements\n")
    report_parts.append("- Recommendations for practical applications\n")
    
    # Write report
    try:
        output_path = os.path.join(os.path.dirname(__file__) or '.', report_filename)
        with open(output_path, "w") as f:
            f.write("".join(report_parts))
        print(f"\nSphere approximation report generated: {os.path.abspath(output_path)}")
    except IOError as e:
        print(f"Error writing report to {report_filename}: {e}")

def main():
    """Main function to run sphere approximation tests."""
    print("Starting Sphere Approximation Volume Estimation Tests")
    print("=" * 60)
    
    # Check if mesh directory exists
    if not os.path.exists(MESH_DIR):
        print(f"Error: Mesh directory '{MESH_DIR}' not found.")
        print("Please ensure the mesh directory is correctly configured.")
        return
    
    # Run tests
    results = run_sphere_approximation_tests()
    
    if results:
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print(f"Results saved to: {VISUALIZATION_OUTPUT_DIR}")
        print("Check the generated markdown report for detailed analysis.")
    else:
        print("No tests were completed successfully.")

if __name__ == "__main__":
    main()

# --- Debug/Test Function ---
def test_single_mesh_sphere_approximation(mesh_filename="bunny.obj", radius_ratio=0.1, max_spheres=10):
    """
    Test sphere approximation on a single mesh for debugging.
    """
    print(f"\n=== DEBUGGING SPHERE APPROXIMATION FOR {mesh_filename} ===")
    
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
        print(f"  Volume: {mesh.volume:.6f}")
        print(f"  Extents: {mesh.extents}")
        
        radius = mesh.extents.max() * radius_ratio
        mesh_name = os.path.splitext(mesh_filename)[0]
        
        # Test both algorithms with visualization enabled
        global ENABLE_VISUALIZATION
        ENABLE_VISUALIZATION = True
        
        print(f"\nTesting Greedy Algorithm...")
        greedy_spheres, greedy_coverage, greedy_time, _ = greedy_sphere_placement(
            mesh, radius, max_spheres, mesh_name
        )
        
        print(f"\nTesting Optimization Algorithm...")
        opt_spheres, opt_coverage, opt_time, _ = optimization_sphere_placement(
            mesh, radius, max_spheres, mesh_name
        )
        
        print(f"\n=== RESULTS COMPARISON ===")
        print(f"Greedy: {len(greedy_spheres)} spheres, {greedy_coverage:.2f}% coverage, {greedy_time:.4f}s")
        print(f"Optimization: {len(opt_spheres)} spheres, {opt_coverage:.2f}% coverage, {opt_time:.4f}s")
        
        if greedy_coverage > opt_coverage:
            print("Winner: Greedy Algorithm")
        elif opt_coverage > greedy_coverage:
            print("Winner: Optimization Algorithm")
        else:
            print("Result: Tie")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

# Uncomment the line below to run a quick test on a single mesh:
# test_single_mesh_sphere_approximation("bunny.obj", 0.1, 10)
