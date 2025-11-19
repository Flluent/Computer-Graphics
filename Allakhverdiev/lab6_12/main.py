# lab6_poly.py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

def to_homogeneous(points):
    pts = np.asarray(points, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    return np.hstack([pts, ones])

def from_homogeneous(points_h):
    ph = np.asarray(points_h, dtype=float)
    w = ph[:, 3:4]
    with np.errstate(divide='ignore', invalid='ignore'):
        coords = ph[:, :3] / w
    return coords

def mat_identity():
    return np.eye(4, dtype=float)

def mat_translate(tx, ty, tz):
    M = mat_identity()
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M

def mat_scale(sx, sy, sz):
    M = mat_identity()
    M[0,0] = sx
    M[1,1] = sy
    M[2,2] = sz
    return M

def mat_scale_about_point(sx, sy, sz, point):
    px, py, pz = point
    return mat_translate(px, py, pz) @ mat_scale(sx, sy, sz) @ mat_translate(-px, -py, -pz)

def mat_rotate_x(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[1,1] = c; M[1,2] = -s
    M[2,1] = s; M[2,2] = c
    return M

def mat_rotate_y(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0,0] = c; M[0,2] = s
    M[2,0] = -s; M[2,2] = c
    return M

def mat_rotate_z(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0,0] = c; M[0,1] = -s
    M[1,0] = s; M[1,1] = c
    return M

def mat_rotate_axis(axis, angle_deg):
    ux, uy, uz = np.asarray(axis, dtype=float)
    norm = math.sqrt(ux*ux + uy*uy + uz*uz)
    if norm == 0:
        return mat_identity()
    ux, uy, uz = ux/norm, uy/norm, uz/norm
    a = math.radians(angle_deg)
    c = math.cos(a); s = math.sin(a)
    R = np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s, 0],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c),    0],
        [0,0,0,1]
    ], dtype=float)
    return R

class Point3D:
    def __init__(self, x, y, z):
        self.coords = np.array([float(x), float(y), float(z)], dtype=float)
    def as_array(self):
        return self.coords.copy()
    def transform(self, M):
        p_h = to_homogeneous(self.coords.reshape(1,3))
        p2 = (p_h @ M.T)
        return Point3D(*from_homogeneous(p2)[0])
    def __repr__(self):
        return f"Point3D({self.coords[0]:.3f}, {self.coords[1]:.3f}, {self.coords[2]:.3f})"

class Polygon3D:
    def __init__(self, vertices):
        processed = []
        for v in vertices:
            if isinstance(v, Point3D):
                processed.append(v)
            else:
                processed.append(Point3D(v[0], v[1], v[2]))
        self.vertices = processed
    def to_array(self):
        return np.array([v.as_array() for v in self.vertices])
    def transform(self, M):
        arr = self.to_array()
        arr_h = to_homogeneous(arr)
        arr2 = from_homogeneous(arr_h @ M.T)
        return Polygon3D([Point3D(*p) for p in arr2])
    def __len__(self):
        return len(self.vertices)
    def __repr__(self):
        return f"Polygon3D({self.to_array().tolist()})"

class Polyhedron:
    def __init__(self, polygons=None):
        self.polygons = polygons if polygons is not None else []
    def vertices_array(self):
        pts = []
        for poly in self.polygons:
            pts.extend(poly.to_array())
        return np.array(pts)
    def centroid(self):
        arr = self.vertices_array()
        if arr.size == 0:
            return np.array([0.0,0.0,0.0])
        return arr.mean(axis=0)
    def transform(self, M):
        return Polyhedron([poly.transform(M) for poly in self.polygons])
    def apply_inplace(self, M):
        self.polygons = [poly.transform(M) for poly in self.polygons]
    def plot(self, ax=None, face_color=(0.7,0.8,1.0), edge_color='k', alpha=0.9, linewidth=1):
        if ax is None:
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False
        poly_verts = [poly.to_array() for poly in self.polygons]
        coll = Poly3DCollection(poly_verts, facecolors=face_color, edgecolors=edge_color,
                                linewidths=linewidth, alpha=alpha)
        ax.add_collection3d(coll)
        all_pts = self.vertices_array()
        if all_pts.size == 0:
            return ax
        xmin, ymin, zmin = all_pts.min(axis=0)
        xmax, ymax, zmax = all_pts.max(axis=0)
        max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
        cx = 0.5*(xmax + xmin)
        cy = 0.5*(ymax + ymin)
        cz = 0.5*(zmax + zmin)
        half = max_range/2 + 1e-6
        ax.set_xlim(cx-half, cx+half)
        ax.set_ylim(cy-half, cy+half)
        ax.set_zlim(cz-half, cz+half)
        ax.set_box_aspect([1,1,1])
        if created_fig:
            plt.show()
        return ax

def make_tetrahedron():
    a = 1.0
    verts = [
        ( a,  a,  a),
        ( a, -a, -a),
        (-a,  a, -a),
        (-a, -a,  a),
    ]
    faces = [
        [0,1,2],
        [0,3,1],
        [0,2,3],
        [1,3,2],
    ]
    polys = []
    for f in faces:
        polys.append(Polygon3D([Point3D(*verts[i]) for i in f]))
    return Polyhedron(polys)

def make_cube():
    v = [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
         (-1,-1,1),  (1,-1,1),  (1,1,1),  (-1,1,1)]
    faces_idx = [
        [0,1,2,3],
        [4,7,6,5],
        [0,4,5,1],
        [1,5,6,2],
        [2,6,7,3],
        [3,7,4,0],
    ]
    polys = []
    for f in faces_idx:
        polys.append(Polygon3D([Point3D(*v[i]) for i in f]))
    return Polyhedron(polys)

def make_octahedron():
    verts = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    faces_idx = [
        [0,2,4], [2,1,4], [1,3,4], [3,0,4],
        [2,0,5], [1,2,5], [3,1,5], [0,3,5]
    ]
    polys = []
    for f in faces_idx:
        polys.append(Polygon3D([Point3D(*verts[i]) for i in f]))
    return Polyhedron(polys)

# 5-6: Функции для вращения вокруг произвольной оси и выбора проекции
def mat_rotate_about_axis(p1, p2, angle_deg):
    """
    Матрица поворота вокруг произвольной оси, заданной двумя точками p1 и p2
    """
    axis = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return mat_identity()
    axis = axis / norm
    T1 = mat_translate(-p1[0], -p1[1], -p1[2])
    T2 = mat_translate(p1[0], p1[1], p1[2])
    R = mat_rotate_axis(axis, angle_deg)
    return T2 @ R @ T1

def mat_rotate_about_center_axis(polyhedron, axis_direction, angle_deg):
    """
    Матрица поворота вокруг прямой через центр многогранника параллельно координатной оси
    """
    center = polyhedron.centroid()
    if axis_direction == 'x':
        p2 = center + np.array([1, 0, 0])
    elif axis_direction == 'y':
        p2 = center + np.array([0, 1, 0])
    elif axis_direction == 'z':
        p2 = center + np.array([0, 0, 1])
    else:
        raise ValueError("axis_direction must be 'x', 'y' or 'z'")
    return mat_rotate_about_axis(center, p2, angle_deg)

class Polyhedron56(Polyhedron):
    """Расширенный класс многогранника с дополнительными преобразованиями для заданий 5-6"""
    
    def rotate_about_axis(self, p1, p2, angle_deg):
        """Поворот вокруг произвольной оси (по двум точкам)"""
        M = mat_rotate_about_axis(p1, p2, angle_deg)
        return Polyhedron56([poly.transform(M) for poly in self.polygons])
    
    def rotate_about_center_axis(self, axis_direction, angle_deg):
        """Поворот вокруг прямой через центр, параллельной координатной оси"""
        M = mat_rotate_about_center_axis(self, axis_direction, angle_deg)
        return Polyhedron56([poly.transform(M) for poly in self.polygons])
    
    def plot_with_projection(self, ax=None, projection_type='perspective', **kwargs):
        """
        Построение с выбором типа проекции
        projection_type: 'perspective' или 'axonometric'
        """
        if ax is None:
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False
        
        if projection_type == 'axonometric':
            ax.set_proj_type('ortho')
        else:
            ax.set_proj_type('persp')
        
        super().plot(ax=ax, **kwargs)
        
        if created_fig:
            plt.show()
        return ax

def make_polyhedron56_tetrahedron():
    base = make_tetrahedron()
    return Polyhedron56(base.polygons)

def make_polyhedron56_cube():
    base = make_cube()
    return Polyhedron56(base.polygons)

def make_polyhedron56_octahedron():
    base = make_octahedron()
    return Polyhedron56(base.polygons)

# 5-6: Демонстрация нового функционала
def demo56():
    """Демонстрация функционала для заданий 5-6"""
    
    tet = make_polyhedron56_tetrahedron()
    cube = make_polyhedron56_cube()
    octa = make_polyhedron56_octahedron()
    
    # Демонстрация разных проекций
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    cube.plot_with_projection(ax=ax1, projection_type='perspective', 
                             face_color=(0.7, 0.8, 1.0))
    ax1.set_title("Перспективная проекция")
    
    ax2 = fig.add_subplot(122, projection='3d')
    cube.plot_with_projection(ax=ax2, projection_type='axonometric', 
                             face_color=(0.8, 1.0, 0.7))
    ax2.set_title("Аксонометрическая проекция")
    
    plt.suptitle("Сравнение типов проекций")
    plt.show()
    
    # Вращение вокруг оси через центр, параллельной координатной оси
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    cube.plot_with_projection(ax=ax1, face_color=(0.7, 0.9, 0.7))
    ax1.set_title("Исходный куб")
    
    cube_rotated_y = cube.rotate_about_center_axis('y', 45)
    ax2 = fig.add_subplot(132, projection='3d')
    cube_rotated_y.plot_with_projection(ax=ax2, face_color=(0.9, 0.7, 0.7))
    ax2.set_title("Поворот на 45° вокруг оси Y через центр")
    
    cube_rotated_z = cube.rotate_about_center_axis('z', 30)
    ax3 = fig.add_subplot(133, projection='3d')
    cube_rotated_z.plot_with_projection(ax=ax3, face_color=(0.7, 0.7, 0.9))
    ax3.set_title("Поворот на 30° вокруг оси Z через центр")
    
    plt.suptitle("Вращение вокруг осей через центр многогранника")
    plt.show()
    
    # Поворот вокруг произвольной оси (по двум точкам)
    fig = plt.figure(figsize=(12, 6))
    
    p1 = np.array([-2, -2, -2])
    p2 = np.array([2, 2, 2])
    
    ax1 = fig.add_subplot(121, projection='3d')
    octa.plot_with_projection(ax=ax1, face_color=(0.8, 0.8, 1.0))
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
             'r-', linewidth=3, label='Ось вращения')
    ax1.legend()
    ax1.set_title("Исходный октаэдр с осью вращения")
    
    octa_rotated = octa.rotate_about_axis(p1, p2, 60)
    ax2 = fig.add_subplot(122, projection='3d')
    octa_rotated.plot_with_projection(ax=ax2, face_color=(1.0, 0.8, 0.8))
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
             'r-', linewidth=3, label='Ось вращения')
    ax2.legend()
    ax2.set_title("Октаэдр после поворота на 60° вокруг произвольной оси")
    
    plt.suptitle("Поворот вокруг произвольной прямой")
    plt.show()

def demo():
    tet = make_tetrahedron()
    cube = make_cube()
    octa = make_octahedron()

    fig = plt.figure(figsize=(15,5))
    axs = [fig.add_subplot(1,3,i+1, projection='3d') for i in range(3)]
    tet.plot(ax=axs[0], face_color=(1.0,0.6,0.6)); axs[0].set_title("Tetrahedron")
    cube.plot(ax=axs[1], face_color=(0.6,1.0,0.6)); axs[1].set_title("Cube")
    octa.plot(ax=axs[2], face_color=(0.6,0.6,1.0)); axs[2].set_title("Octahedron")
    plt.suptitle("Исходные правильные многогранники")
    plt.show()

    tx, ty, tz = 2.0, 0.5, -1.0
    sx, sy, sz = 0.8, 1.5, 0.6
    angle = 40

    center = cube.centroid()
    M_scale_center = mat_scale_about_point(sx, sy, sz, center)
    M_rotate = mat_rotate_axis((1,1,0), angle)
    M_translate = mat_translate(tx, ty, tz)
    M = M_translate @ M_rotate @ M_scale_center

    transformed_cube = cube.transform(M)

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    cube.plot(ax=ax1, face_color=(0.7,0.9,0.7)); ax1.set_title("Исходный куб")
    ax2 = fig.add_subplot(122, projection='3d')
    transformed_cube.plot(ax=ax2, face_color=(0.9,0.7,0.7)); ax2.set_title(
        f"Куб после преобразований")
    plt.show()

    M1 = mat_rotate_z(30)
    M2 = mat_translate(0.5, 0.0, 0.2)
    M3 = mat_scale(1.2, 1.2, 1.2)
    M_total = M3 @ M2 @ M1
    transformed_tet = tet.transform(M_total)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121, projection='3d')
    tet.plot(ax=ax, face_color=(1,0.8,0.6)); ax.set_title("Исходный тетраэдр")
    ax2 = fig.add_subplot(122, projection='3d')
    transformed_tet.plot(ax=ax2, face_color=(0.6,0.8,1.0)); ax2.set_title("Тетраэдр после преобразований")
    plt.show()

if __name__ == "__main__":
    print("=== ДЕМО ОСНОВНЫХ ВОЗМОЖНОСТЕЙ ===")
    demo()
    
    print("\n=== ДЕМО ФУНКЦИОНАЛА 5-6 ===")
    demo56()