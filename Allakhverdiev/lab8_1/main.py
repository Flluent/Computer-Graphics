import cv2
import numpy as np
import math

# Define a simple 3D cube object
# Vertices of the cube
vertices = np.array([
    [-1, -1, -1],  # 0
    [1, -1, -1],  # 1
    [1, 1, -1],  # 2
    [-1, 1, -1],  # 3
    [-1, -1, 1],  # 4
    [1, -1, 1],  # 5
    [1, 1, 1],  # 6
    [-1, 1, 1]  # 7
], dtype=np.float32)

# Faces of the cube (each face is a list of vertex indices)
faces = [
    [0, 1, 2, 3],  # Back
    [4, 5, 6, 7],  # Front
    [0, 1, 5, 4],  # Bottom
    [3, 2, 6, 7],  # Top
    [0, 3, 7, 4],  # Left
    [1, 2, 6, 5]  # Right
]

# Compute outward normals for each face
normals = []
for face in faces:
    # Take two vectors in the plane of the face
    v1 = vertices[face[1]] - vertices[face[0]]
    v2 = vertices[face[2]] - vertices[face[1]]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)  # Normalize
    # Ensure outward direction (for cube, we can check dot with center to vertex, but since it's convex, we assume order is CCW for outward)
    normals.append(normal)

# Rotation angles
angle_x = 0
angle_y = 0
angle_z = 0

# View vector (from camera to object, normalized)
view_vector = np.array([0, 0, -1], dtype=np.float32)  # Initially looking along -Z

# Projection mode: 0 - parallel (orthographic), 1 - perspective
projection_mode = 0

# Camera position for perspective (distance along view vector)
camera_distance = 5.0


# Function to get rotation matrix
def get_rotation_matrix(ax, ay, az):
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)]
    ])
    ry = np.array([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])
    rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx


# Function to project points
def project_points(points, rot_matrix, mode):
    rotated = (rot_matrix @ points.T).T
    if mode == 0:  # Parallel (orthographic)
        projected = rotated[:, :2] * 200 + np.array([400, 300])  # Scale and center
    else:  # Perspective
        # Translate along view vector to camera position
        rotated[:, 2] += camera_distance
        # Perspective divide
        projected = np.zeros((len(points), 2))
        for i, p in enumerate(rotated):
            if p[2] > 0:  # Avoid division by zero or negative
                projected[i] = p[:2] / p[2] * 200 + np.array([400, 300])
            else:
                projected[i] = [0, 0]  # Clip
    return projected.astype(np.int32)


# Function to draw the object
def draw_object(img, vertices, faces, normals, rot_matrix, view_vec, mode):
    projected = project_points(vertices, rot_matrix, mode)

    # For each face
    for i, face in enumerate(faces):
        # Compute face normal in world space
        face_normal = (rot_matrix @ normals[i]).flatten()

        # Backface culling: dot product of normal and view vector
        # View vector should be from face to camera, so if dot > 0, facing camera (assuming view_vec points towards object)
        # Typically, view_vec is from camera to object, so dot(normal, -view_vec) > 0 means front
        if np.dot(face_normal, -view_vec) < 0:
            continue  # Cull backface

        # Draw the face
        pts = projected[face]
        cv2.fillConvexPoly(img, pts, (100, 100, 255))  # Fill with color
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)  # Outline


# Main loop
window_name = "Rotating Cube with Backface Culling"
cv2.namedWindow(window_name)

while True:
    img = np.zeros((600, 800, 3), np.uint8)

    rot_matrix = get_rotation_matrix(math.radians(angle_x), math.radians(angle_y), math.radians(angle_z))

    draw_object(img, vertices, faces, normals, rot_matrix, view_vector, projection_mode)

    # Display instructions
    cv2.putText(img, "Press X/Y/Z to rotate, V to change view, P to toggle projection, Q to quit", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Projection: {'Parallel' if projection_mode == 0 else 'Perspective'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('x'):
        angle_x = (angle_x + 5) % 360
    elif key == ord('y'):
        angle_y = (angle_y + 5) % 360
    elif key == ord('z'):
        angle_z = (angle_z + 5) % 360
    elif key == ord('v'):
        # Change view vector, for simplicity, rotate it around Y axis
        view_angle = math.atan2(view_vector[0], view_vector[2]) + math.radians(15)
        view_vector[0] = math.sin(view_angle)
        view_vector[2] = math.cos(view_angle)
        view_vector /= np.linalg.norm(view_vector)
    elif key == ord('p'):
        projection_mode = 1 - projection_mode  # Toggle

cv2.destroyAllWindows()