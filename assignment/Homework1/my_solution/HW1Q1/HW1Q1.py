import open3d as o3d
import numpy as np

# 1. Define ellipsoid parameters
a = 1.0
b = 1.0
c = 0.5

# 2. Define the mapping function f(u, v) from parameter space to 3D
def f(u, v):
    """Maps (u, v) coordinates to 3D points on the ellipsoid."""
    # Ensure inputs are numpy arrays for broadcasting
    u, v = np.asarray(u), np.asarray(v)
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    # Stack and transpose to get shape (..., 3)
    return np.stack([x, y, z], axis=-1)

# 3. Generate ellipsoid mesh geometry
N_u = 60  # Resolution along u-direction
N_v = 30  # Resolution along v-direction
u_range = np.linspace(-np.pi, np.pi, N_u)
v_range = np.linspace(0, np.pi, N_v)
uu, vv = np.meshgrid(u_range, v_range) # uu: (N_v, N_u), vv: (N_v, N_u)

# Calculate vertex positions by applying f to the grid
vertices = f(uu.flatten(), vv.flatten()) # Shape (N_v * N_u, 3)

# Generate triangles connecting the grid vertices
triangles = []
for i in range(N_v - 1):
    for j in range(N_u - 1):
        # Vertex indices for the current quad
        idx00 = i * N_u + j
        idx10 = i * N_u + (j + 1)
        idx01 = (i + 1) * N_u + j
        idx11 = (i + 1) * N_u + (j + 1)
        # Create two triangles for the quad with reversed winding order
        triangles.append([idx00, idx01, idx10])
        triangles.append([idx10, idx01, idx11])

# Create Open3D TriangleMesh object
ellipsoid_mesh = o3d.geometry.TriangleMesh()
ellipsoid_mesh.vertices = o3d.utility.Vector3dVector(vertices)
ellipsoid_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
ellipsoid_mesh.compute_vertex_normals() # Compute normals for smooth shading
ellipsoid_mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Set color to gray


# --- Calculations for Principal Directions (Q4) ---
p_uv = (np.pi / 4, np.pi / 6) # Point p in parameter space
u0, v0 = p_uv

# Calculate the point f(p) on the ellipsoid surface
point_on_surface = f(u0, v0).flatten() # Ensure it's a 1D array/vector
print(f"Point f(p) on surface: {point_on_surface}")

# Calculate tangent vectors t_u, t_v (which are the principal directions for this parameterization)
t_u = np.array([
    -a * np.sin(u0) * np.sin(v0),
     b * np.cos(u0) * np.sin(v0),
     0
])
t_v = np.array([
     a * np.cos(u0) * np.cos(v0),
     b * np.sin(u0) * np.cos(v0),
    -c * np.sin(v0)
])
print(f"Principal Direction 1 (t_u): {t_u}")
print(f"Principal Direction 2 (t_v): {t_v}")

# Optional: Calculate unit normal vector for context
# Note: The normal calculated from t_u x t_v might point inwards or outwards
# depending on the order. Ensure it matches desired convention if needed.
normal_vec_unscaled = np.cross(t_u, t_v)
normal_norm = np.linalg.norm(normal_vec_unscaled)
if not np.isclose(normal_norm, 0.0):
    normal_vector = normal_vec_unscaled / normal_norm
    print(f"Unit Normal vector n: {normal_vector}")
else:
    normal_vector = None
    print("Normal vector is zero (degenerate point).")


# --- Visualization Setup ---
vis_elements = [ellipsoid_mesh]

# Visualize the point f(p) using a small sphere
point_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
point_marker.translate(point_on_surface)
point_marker.paint_uniform_color([0, 0, 1]) # Blue color
vis_elements.append(point_marker)

# Helper function to create an arrow for visualization
def create_arrow(origin, vector, scale=0.3, color=[1, 0, 0]):
    """Creates an Open3D arrow mesh."""
    vec_norm = np.linalg.norm(vector)
    if np.isclose(vec_norm, 0.0): return None # Avoid division by zero or zero-length arrow
    unit_vec = vector / vec_norm
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=max(0.005, 0.01 * scale), # Ensure minimum radius
        cone_radius=max(0.01, 0.02 * scale),
        cylinder_height=0.8 * scale * vec_norm, # Adjust length based on vector magnitude
        cone_height=0.2 * scale * vec_norm,
        resolution=20
    )
    # Compute rotation matrix to align arrow (initially pointing Z+) with the vector
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_theta = np.dot(z_axis, unit_vec)
    # Handle identical or opposite vectors
    if np.isclose(cos_theta, 1.0): # Aligned with Z+
        pass # No rotation needed
    elif np.isclose(cos_theta, -1.0): # Opposite to Z+
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0] * np.pi)
        arrow.rotate(R, center=[0,0,0])
    else:
        rotation_axis = np.cross(z_axis, unit_vec)
        # Normalize rotation axis if it's not zero
        if not np.isclose(np.linalg.norm(rotation_axis), 0.0):
            rotation_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # Clip for safety
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            arrow.rotate(R, center=[0,0,0])
        # Else: vector is parallel to z-axis, handled above or no rotation needed

    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow

# 4c) Visualize Principal Directions t_u and t_v
arrow_pd1 = create_arrow(point_on_surface, t_u, scale=0.5, color=[1, 0, 0]) # Red for t_u
arrow_pd2 = create_arrow(point_on_surface, t_v, scale=0.5, color=[0, 1, 0]) # Green for t_v
if arrow_pd1: vis_elements.append(arrow_pd1)
if arrow_pd2: vis_elements.append(arrow_pd2)


# Optional: Visualize the normal vector as a cyan arrow
if normal_vector is not None:
    normal_arrow = create_arrow(point_on_surface, normal_vector, scale=0.3, color=[0, 1, 1]) # Cyan
    if normal_arrow: vis_elements.append(normal_arrow)


# Add coordinate axes for reference
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
vis_elements.append(coordinate_frame)

# --- Run Visualization ---
print("\nVisualizing ellipsoid, f(p), and Principal Directions.")
print("Point f(p) = Blue sphere")
print("Principal Direction 1 (t_u) = Red arrow")
print("Principal Direction 2 (t_v) = Green arrow")
print("Normal n = Cyan arrow (optional)")
print("Close the visualization window to exit.")

o3d.visualization.draw_geometries(
    vis_elements,
    window_name="Principal Curvature Directions (HW1 Q4)"
)

print("Visualization finished.")