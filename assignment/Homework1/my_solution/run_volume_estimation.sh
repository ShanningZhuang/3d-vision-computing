#!/bin/bash

# Mesh Volume Estimation - Runner Script
# This script installs the required dependencies and runs the volume estimation code

echo "=== 3D Mesh Volume Estimation ==="
echo "Installing required dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install required packages
pip install numpy trimesh pyvista matplotlib memory_profiler psutil

# Run the volume estimation script
echo -e "\nRunning volume estimation..."
python HW1Q4_volume_estimation.py

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo -e "\nVolume estimation completed successfully!"
    echo "Results have been saved to the current directory."
    echo "See HW1Q4_report.md for a detailed analysis of the results."
else
    echo -e "\nError: Volume estimation failed. Please check the error messages above."
fi 