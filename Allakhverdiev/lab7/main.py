"""
ЛР-7. Загрузка и сохранение модели многогранника из файла (Wavefront OBJ)
Основано на ЛР-6: правильные многогранники + аффинные преобразования (матрицы 4x4)

Добавлено:
 - Загрузка .obj модели (Polyhedron.from_obj)
 - Сохранение модели в .obj (Polyhedron.to_obj)
 - Отображение загруженной модели
 - Применение аффинных преобразований
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

# =====================================================
# Утилиты для однородных координат
# =====================================================
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

# =====================================================
# Матрицы 4x4 (аффинные преобразования)
# =====================================================
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
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M

def mat_scale_about_point(sx, sy, sz, point):
    px, py, pz = point
    return mat_translate(px, py, pz) @ mat_scale(sx, sy, sz) @ mat_translate(-px, -py, -pz)

def mat_rotate_x(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[1, 1] = c; M[1, 2] = -s
    M[2, 1] = s; M[2, 2] = c
    return M

def mat_rotate_y(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0, 0] = c; M[0, 2] = s
    M[2, 0] = -s; M[2, 2] = c
    return M

def mat_rotate_z(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0, 0] = c; M[0, 1] = -s
    M[1, 0] = s; M[1, 1] = c
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
        [0, 0, 0, 1]
    ], dtype=float)
    return R

# =====================================================
# Классы геометрии
# =====================================================
class Point3D:
    def __init__(self, x, y, z):
        self.coords = np.array([float(x), float(y), float(z)], dtype=float)

    def as_array(self):
        return self.coords.copy()

    def transform(self, M):
        p_h = to_homogeneous(self.coords.reshape(1, 3))
        p2 = (p_h @ M.T)
        return Point3D(*from_homogeneous(p2)[0])

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
            return np.array([0.0, 0.0, 0.0])
        return arr.mean(axis=0)

    def transform(self, M):
        return Polyhedron([poly.transform(M) for poly in self.polygons])

    def apply_inplace(self, M):
        self.polygons = [poly.transform(M) for poly in self.polygons]

    def plot(self, ax=None, face_color=(0.7, 0.8, 1.0), edge_color='k', alpha=0.9, linewidth=1):
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False

        poly_verts = [poly.to_array() for poly in self.polygons]
        coll = Poly3DCollection(poly_verts, facecolors=face_color, edgecolors=edge_color,
                                linewidths=linewidth, alpha=alpha)
        ax.add_collection3d(coll)

        all_pts = self.vertices_array()
        if all_pts.size > 0:
            xmin, ymin, zmin = all_pts.min(axis=0)
            xmax, ymax, zmax = all_pts.max(axis=0)
            max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
            cx, cy, cz = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
            half = max_range / 2 + 1e-6
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_zlim(cz - half, cz + half)
            ax.set_box_aspect([1, 1, 1])
        if created_fig:
            plt.show()
        return ax

    # =====================================================
    # Загрузка и сохранение .OBJ
    # =====================================================
    @staticmethod
    def from_obj(filepath):
        """Загрузка модели из Wavefront OBJ"""
        vertices = []
        faces = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = map(float, parts[1:4])
                    vertices.append((x, y, z))
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    idx = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(idx)
        polygons = []
        for face in faces:
            polygons.append(Polygon3D([Point3D(*vertices[i]) for i in face]))
        return Polyhedron(polygons)

    def to_obj(self, filepath):
        """Сохранение модели в формате OBJ"""
        verts = []
        faces = []
        v_counter = 1
        for poly in self.polygons:
            arr = poly.to_array()
            verts.extend(arr)
            faces.append(list(range(v_counter, v_counter + len(arr))))
            v_counter += len(arr)
        with open(filepath, 'w', encoding='utf-8') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write("f " + " ".join(map(str, face)) + "\n")

# =====================================================
# Демонстрация работы
# =====================================================
def demo_obj_load():
    path = "utah_teapot_lowpoly.obj"  # пример OBJ из задания
    model = Polyhedron.from_obj(path)
    print(f"Загружено граней: {len(model.polygons)}")

    # Исходная модель
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    model.plot(ax=ax1, face_color=(0.6, 0.8, 1.0))
    ax1.set_title("Исходная модель OBJ")

    # Преобразуем: поворот + масштаб + перенос
    M = mat_translate(0, 0, 2) @ mat_rotate_y(30) @ mat_scale(1.2, 1.2, 1.2)
    model_t = model.transform(M)

    ax2 = fig.add_subplot(122, projection='3d')
    model_t.plot(ax=ax2, face_color=(1.0, 0.6, 0.6))
    ax2.set_title("После преобразований")
    plt.show()

    # Сохраняем обратно
    model_t.to_obj("teapot_transformed.obj")
    print("Сохранено в teapot_transformed.obj")

if __name__ == "__main__":
    demo_obj_load()
