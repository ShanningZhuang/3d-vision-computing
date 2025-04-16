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

def create_velocity_vector(p, v, a=1, b=1, c=0.5, scale=0.2):
    """Create a vector to represent velocity at point p."""
    u0, v0 = p
    du, dv = v
    
    # Starting point (point p on ellipsoid)
    start_point = ellipsoid_function(u0, v0, a, b, c)
    
    # Calculate tangent at point p
    epsilon = 1e-5
    p_plus = ellipsoid_function(u0 + epsilon * du, v0 + epsilon * dv, a, b, c)
    tangent = (p_plus - start_point) / epsilon
    
    # Normalize and scale the tangent vector
    tangent = tangent / np.linalg.norm(tangent) * scale
    
    # End point
    end_point = start_point + tangent
    
    # Create line set for the vector
    points = [start_point, end_point]
    lines = [[0, 1]]
    colors = [[0, 0, 1]]  # Blue color
    
    vector = o3d.geometry.LineSet()
    vector.points = o3d.utility.Vector3dVector(points)
    vector.lines = o3d.utility.Vector2iVector(lines)
    vector.colors = o3d.utility.Vector3dVector(colors)
    
    return vector

def create_tangent_plane(p, a=1, b=1, c=0.5, size=0.4):
    """Create a tangent plane at point p."""
    u0, v0 = p
    point = ellipsoid_function(u0, v0, a, b, c)
    
    # Calculate normal vector at point p (gradient of ellipsoid function)
    normal = np.array([
        point[0] / a**2,
        point[1] / b**2,
        point[2] / c**2
    ])
    normal = normal / np.linalg.norm(normal)
    
    # Find two orthogonal vectors in the tangent plane
    if not np.isclose(normal[0], 0):
        v1 = np.array([(-normal[1] - normal[2]) / normal[0], 1, 1])
    elif not np.isclose(normal[1], 0):
        v1 = np.array([1, (-normal[0] - normal[2]) / normal[1], 1])
    else:
        v1 = np.array([1, 1, (-normal[0] - normal[1]) / normal[2]])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Create corners of the tangent plane
    corners = [
        point + size * (-v1 - v2),
        point + size * (v1 - v2),
        point + size * (v1 + v2),
        point + size * (-v1 + v2)
    ]
    
    # Create triangle mesh for the tangent plane
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(corners)
    plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    
    # Set color and transparency
    plane.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue
    
    return plane, v1, v2

def create_basis_vectors(p, v1, v2, a=1, b=1, c=0.5, scale=0.2):
    """Create two orthogonal basis vectors in the tangent plane."""
    point = ellipsoid_function(p[0], p[1], a, b, c)
    
    # Create line sets for the basis vectors
    basis1_points = [point, point + scale * v1]
    basis2_points = [point, point + scale * v2]
    
    # First basis vector (cyan)
    basis1 = o3d.geometry.LineSet()
    basis1.points = o3d.utility.Vector3dVector(basis1_points)
    basis1.lines = o3d.utility.Vector2iVector([[0, 1]])
    basis1.colors = o3d.utility.Vector3dVector([[0, 1, 1]])  # Cyan
    
    # Second basis vector (magenta)
    basis2 = o3d.geometry.LineSet()
    basis2.points = o3d.utility.Vector3dVector(basis2_points)
    basis2.lines = o3d.utility.Vector2iVector([[0, 1]])
    basis2.colors = o3d.utility.Vector3dVector([[1, 0, 1]])  # Magenta
    
    return basis1, basis2

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
    
    # Create velocity vector at point p
    velocity_vector = create_velocity_vector(p, v)
    
    # Create tangent plane and basis vectors
    tangent_plane, v1, v2 = create_tangent_plane(p)
    basis1, basis2 = create_basis_vectors(p, v1, v2)
    
    # Visualize
    o3d.visualization.draw_geometries([mesh, line_set, sphere, velocity_vector, 
                                      tangent_plane, basis1, basis2])
