import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def ellipsoid_function(u, v, a=1, b=1, c=0.5):
    """Compute the 3D coordinates on ellipsoid for given u, v parameters."""
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    return np.array([x, y, z])

def create_ellipsoid_mesh(a=1, b=1, c=0.5, resolution=50):
    """Create a triangle mesh of an ellipsoid."""
    u = np.linspace(-np.pi, np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    vertices = []
    triangles = []
    
    # Create vertices
    for i in range(resolution):
        for j in range(resolution):
            vertices.append(ellipsoid_function(u[i], v[j], a, b, c))
    
    # Create triangles
    for i in range(resolution-1):
        for j in range(resolution-1):
            idx = i * resolution + j
            triangles.append([idx, idx + 1, idx + resolution])
            triangles.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    # Vector3dVector: 3D vector of double
    # Vector3iVector: 3D vector of int
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    
    return mesh

def create_curve_on_ellipsoid(p, v, a=1, b=1, c=0.5, t_range=1, num_points=100):
    """Create a curve on the ellipsoid passing through point p with tangent v."""
    # Parameter point and direction
    u0, v0 = p
    du, dv = v
    
    # Sample t values
    t_values = np.linspace(-t_range, t_range, num_points)
    
    # Compute curve points
    curve_points = []
    for t in t_values:
        u = u0 + t * du
        v = v0 + t * dv
        # Ensure v stays within bounds
        v = np.clip(v, 0, np.pi)
        point = ellipsoid_function(u, v, a, b, c)
        curve_points.append(point)
    
    return np.array(curve_points)

def visualize_curve_as_cylinders(curve_points, radius=0.01):
    """Create a line set to visualize the curve using cylinders."""
    line_segments = [[i, i+1] for i in range(len(curve_points)-1)]
    colors = [[1, 0, 0] for _ in range(len(line_segments))]  # Red color
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(curve_points)
    line_set.lines = o3d.utility.Vector2iVector(line_segments)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

if __name__ == "__main__":
    # Create the ellipsoid mesh
    mesh = create_ellipsoid_mesh()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    
    # Point p and direction v from the problem
    p = (np.pi/4, np.pi/6)
    v = (1, 0)
    
    # Create the curve on the ellipsoid
    curve_points = create_curve_on_ellipsoid(p, v)
    
    # Create visualization objects for the curve
    line_set = visualize_curve_as_cylinders(curve_points)
    
    # Create a sphere to mark the point p on the curve
    point_p_3d = ellipsoid_function(p[0], p[1])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.translate(point_p_3d)
    sphere.paint_uniform_color([0, 1, 0])  # Green
    
    # Visualize
    o3d.visualization.draw_geometries([mesh, line_set, sphere])
