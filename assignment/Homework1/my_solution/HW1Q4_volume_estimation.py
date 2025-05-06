"""
3D Vision Computing - Homework 1
Problem 4 - Sub Problem 1: Mesh Volume Estimation

This script implements two methods for estimating the volume of a 3D mesh:
1. Voxelization: Convert the mesh into voxels and count occupied voxels.
2. Signed Distance Field (SDF) + Monte Carlo: Sample points randomly and check if they're inside the mesh.

Both methods are compared in terms of accuracy, speed, and memory usage.
"""

import os
import time
import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import psutil

def get_mesh_paths():
    """Returns paths to the mesh files."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "objs_approx")
    mesh_paths = {
        "bunny": os.path.join(base_dir, "bunny.obj"),
        "apple": os.path.join(base_dir, "apple.obj"),
        "airplane": os.path.join(base_dir, "airplane.obj")
    }
    return mesh_paths

def estimate_volume_voxelization(mesh, resolution):
    """
    Estimate volume using voxelization.
    
    Args:
        mesh: Trimesh mesh object
        resolution: Resolution of the voxel grid (number of voxels along the longest dimension)
        
    Returns:
        estimated_volume: Estimated volume of the mesh
    """
    # Get the bounding box dimensions
    bounds = mesh.bounds
    bbox_min, bbox_max = bounds[0], bounds[1]
    bbox_dimensions = bbox_max - bbox_min
    
    # Determine voxel size based on the resolution
    voxel_size = max(bbox_dimensions) / resolution
    
    # Create a voxel grid
    voxel_grid = mesh.voxelized(pitch=voxel_size)
    
    # Count filled voxels and calculate volume
    filled_voxels = voxel_grid.filled_count
    voxel_volume = voxel_size**3
    estimated_volume = filled_voxels * voxel_volume
    
    return estimated_volume, voxel_grid

def estimate_volume_sdf_monte_carlo(mesh, num_samples):
    """
    Estimate volume using SDF + Monte Carlo method.
    
    Args:
        mesh: Trimesh mesh object
        num_samples: Number of random points to sample
        
    Returns:
        estimated_volume: Estimated volume of the mesh
    """
    # Get the bounding box
    bounds = mesh.bounds
    bbox_min, bbox_max = bounds[0], bounds[1]
    bbox_dimensions = bbox_max - bbox_min
    bbox_volume = np.prod(bbox_dimensions)
    
    # Convert trimesh to pyvista mesh for SDF computation
    vertices = mesh.vertices
    faces = mesh.faces
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    
    # Generate random points within the bounding box
    random_points = np.random.uniform(
        low=bbox_min,
        high=bbox_max,
        size=(num_samples, 3)
    )
    
    # Compute signed distance for each point
    signed_distances = pv_mesh.compute_implicit_distance(random_points)
    
    # Count points inside the mesh (negative signed distance)
    points_inside = np.sum(signed_distances < 0)
    
    # Estimate volume
    volume_fraction = points_inside / num_samples
    estimated_volume = volume_fraction * bbox_volume
    
    return estimated_volume, points_inside, num_samples

def run_benchmarks():
    """Run benchmarks for different methods and parameters."""
    mesh_paths = get_mesh_paths()
    results = {}
    
    for mesh_name, mesh_path in mesh_paths.items():
        print(f"\nProcessing {mesh_name} mesh...")
        mesh = trimesh.load_mesh(mesh_path)
        mesh_results = {"voxelization": {}, "monte_carlo": {}}
        
        # True volume (approximate reference using high-resolution voxelization)
        print("Computing reference volume with high-resolution voxelization...")
        reference_volume, _ = estimate_volume_voxelization(mesh, 256)
        mesh_results["reference_volume"] = reference_volume
        print(f"Reference volume: {reference_volume:.6f} cubic units")
        
        # Test voxelization with different resolutions
        resolutions = [32, 64, 128, 192]
        for resolution in resolutions:
            print(f"Testing voxelization with resolution {resolution}...")
            
            # Measure execution time
            start_time = time.time()
            mem_usage = memory_usage((estimate_volume_voxelization, (mesh, resolution)), max_iterations=1)
            estimated_volume, _ = estimate_volume_voxelization(mesh, resolution)
            execution_time = time.time() - start_time
            
            # Calculate error
            relative_error = abs(estimated_volume - reference_volume) / reference_volume * 100
            
            mesh_results["voxelization"][resolution] = {
                "volume": estimated_volume,
                "time": execution_time,
                "memory": max(mem_usage) - min(mem_usage),
                "error": relative_error
            }
            
            print(f"  Volume: {estimated_volume:.6f} cubic units")
            print(f"  Time: {execution_time:.4f} seconds")
            print(f"  Memory: {max(mem_usage) - min(mem_usage):.2f} MiB")
            print(f"  Relative Error: {relative_error:.4f}%")
        
        # Test Monte Carlo with different sample sizes
        sample_sizes = [10000, 50000, 100000, 500000]
        for samples in sample_sizes:
            print(f"Testing Monte Carlo with {samples} samples...")
            
            # Measure execution time
            start_time = time.time()
            mem_usage = memory_usage((estimate_volume_sdf_monte_carlo, (mesh, samples)), max_iterations=1)
            estimated_volume, points_inside, total_points = estimate_volume_sdf_monte_carlo(mesh, samples)
            execution_time = time.time() - start_time
            
            # Calculate error
            relative_error = abs(estimated_volume - reference_volume) / reference_volume * 100
            
            mesh_results["monte_carlo"][samples] = {
                "volume": estimated_volume,
                "time": execution_time,
                "memory": max(mem_usage) - min(mem_usage),
                "error": relative_error,
                "points_inside": points_inside,
                "total_points": total_points
            }
            
            print(f"  Volume: {estimated_volume:.6f} cubic units")
            print(f"  Points inside: {points_inside}/{total_points} ({points_inside/total_points*100:.2f}%)")
            print(f"  Time: {execution_time:.4f} seconds")
            print(f"  Memory: {max(mem_usage) - min(mem_usage):.2f} MiB")
            print(f"  Relative Error: {relative_error:.4f}%")
        
        results[mesh_name] = mesh_results
    
    return results

def plot_results(results):
    """Generate comparison plots for the results."""
    for mesh_name, mesh_results in results.items():
        # Extract data for plotting
        voxel_resolutions = list(mesh_results["voxelization"].keys())
        voxel_times = [mesh_results["voxelization"][r]["time"] for r in voxel_resolutions]
        voxel_memories = [mesh_results["voxelization"][r]["memory"] for r in voxel_resolutions]
        voxel_errors = [mesh_results["voxelization"][r]["error"] for r in voxel_resolutions]
        
        mc_samples = list(mesh_results["monte_carlo"].keys())
        mc_times = [mesh_results["monte_carlo"][s]["time"] for s in mc_samples]
        mc_memories = [mesh_results["monte_carlo"][s]["memory"] for s in mc_samples]
        mc_errors = [mesh_results["monte_carlo"][s]["error"] for s in mc_samples]
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot execution time
        axs[0].plot(voxel_resolutions, voxel_times, 'o-', label='Voxelization')
        axs[0].set_xlabel('Resolution')
        axs[0].set_ylabel('Time (seconds)')
        axs[0].set_title('Execution Time vs. Resolution')
        
        ax2 = axs[0].twiny()
        ax2.plot(mc_samples, mc_times, 'x-', color='red', label='Monte Carlo')
        ax2.set_xlabel('Number of Samples')
        ax2.tick_params(axis='x', labelcolor='red')
        
        lines1, labels1 = axs[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot memory usage
        axs[1].plot(voxel_resolutions, voxel_memories, 'o-', label='Voxelization')
        axs[1].set_xlabel('Resolution')
        axs[1].set_ylabel('Memory (MiB)')
        axs[1].set_title('Memory Usage vs. Resolution')
        
        ax3 = axs[1].twiny()
        ax3.plot(mc_samples, mc_memories, 'x-', color='red', label='Monte Carlo')
        ax3.set_xlabel('Number of Samples')
        ax3.tick_params(axis='x', labelcolor='red')
        
        lines1, labels1 = axs[1].get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot relative error
        axs[2].plot(voxel_resolutions, voxel_errors, 'o-', label='Voxelization')
        axs[2].set_xlabel('Resolution')
        axs[2].set_ylabel('Relative Error (%)')
        axs[2].set_title('Relative Error vs. Resolution/Samples')
        
        ax4 = axs[2].twiny()
        ax4.plot(mc_samples, mc_errors, 'x-', color='red', label='Monte Carlo')
        ax4.set_xlabel('Number of Samples')
        ax4.tick_params(axis='x', labelcolor='red')
        
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"HW1Q4_{mesh_name}_volume_estimation_comparison.png", dpi=300)
        plt.close()

def visualize_voxelization(mesh, voxel_grid, mesh_name, resolution):
    """Visualize the mesh and its voxelization."""
    # Create a pyvista plotter
    plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
    
    # Add the original mesh in wireframe
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))
    plotter.add_mesh(pv_mesh, style='wireframe', color='black', line_width=0.5)
    
    # Add the voxel grid
    voxel_points = voxel_grid.points
    voxel_size = voxel_grid.scale
    
    # Create voxel cubes at each occupied voxel
    for point in voxel_points:
        cube = pv.Cube(center=point, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
        plotter.add_mesh(cube, color='blue', opacity=0.3)
    
    # Set up the camera and save the image
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)
    plotter.screenshot(f"HW1Q4_{mesh_name}_voxel_{resolution}.png")
    plotter.close()

def visualize_monte_carlo(mesh, points, inside_mask, mesh_name, num_samples):
    """Visualize the mesh and Monte Carlo sampling."""
    # Create a pyvista plotter
    plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
    
    # Add the original mesh with transparency
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))
    plotter.add_mesh(pv_mesh, color='gray', opacity=0.3)
    
    # Add sampling points, color-coded by inside/outside
    points_cloud = pv.PolyData(points)
    plotter.add_points(points[inside_mask], color='green', point_size=3, render_points_as_spheres=True)
    plotter.add_points(points[~inside_mask], color='red', point_size=2, render_points_as_spheres=True)
    
    # Set up the camera and save the image
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)
    plotter.screenshot(f"HW1Q4_{mesh_name}_monte_carlo_{num_samples}.png")
    plotter.close()

def main():
    """Main function to run the volume estimation benchmarks."""
    print("3D Mesh Volume Estimation - Comparison of Methods")
    
    # Run benchmarks and get results
    results = run_benchmarks()
    
    # Plot the results
    plot_results(results)
    
    # Visualize examples for the bunny mesh
    mesh_paths = get_mesh_paths()
    bunny_mesh = trimesh.load_mesh(mesh_paths["bunny"])
    
    # Visualize voxelization example
    _, voxel_grid = estimate_volume_voxelization(bunny_mesh, 64)
    visualize_voxelization(bunny_mesh, voxel_grid, "bunny", 64)
    
    # Visualize Monte Carlo example
    # Sample points for visualization (using fewer points for clarity)
    bounds = bunny_mesh.bounds
    bbox_min, bbox_max = bounds[0], bounds[1]
    samples = 1000
    random_points = np.random.uniform(low=bbox_min, high=bbox_max, size=(samples, 3))
    
    # Convert to pyvista mesh for SDF computation
    vertices = bunny_mesh.vertices
    faces = bunny_mesh.faces
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    
    # Compute signed distance for visualization
    signed_distances = pv_mesh.compute_implicit_distance(random_points)
    inside_mask = signed_distances < 0
    
    # Visualize Monte Carlo
    visualize_monte_carlo(bunny_mesh, random_points, inside_mask, "bunny", samples)
    
    print("\nVolume estimation complete. Results saved as images.")

if __name__ == "__main__":
    main() 