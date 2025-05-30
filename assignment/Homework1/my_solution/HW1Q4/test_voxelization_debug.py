#!/usr/bin/env python3
"""
Quick test script for debugging voxelization issues.
This script loads a single mesh and visualizes the voxelization process.
"""

import sys
import os

# Add the current directory to the path so we can import HW1Q4SP1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from HW1Q4SP1 import test_single_mesh_voxelization, get_mesh_files, MESH_DIR

def main():
    print("=== Voxelization Debug Tool ===")
    
    # Get available mesh files
    available_meshes = get_mesh_files(MESH_DIR)
    
    if not available_meshes:
        print(f"No mesh files found in {MESH_DIR}")
        return
    
    print(f"Available meshes: {available_meshes}")
    
    # Test with the first available mesh or specify one
    test_mesh = available_meshes[0]  # Change this to test a specific mesh
    test_resolution = 64  # Change this to test different resolutions
    
    print(f"\nTesting voxelization with:")
    print(f"  Mesh: {test_mesh}")
    print(f"  Resolution: {test_resolution}")
    
    # Run the test
    test_single_mesh_voxelization(test_mesh, test_resolution)
    
    print("\nTest completed. Check the './voxelization_debug' directory for visualization outputs.")

if __name__ == "__main__":
    main() 