# lab6_poly.py
"""
ЛР-6: правильные многогранники + аффинные преобразования (матрицы 4x4)
Классы: Point3D, Polygon3D, Polyhedron
Построены: тетраэдр, куб (гексаэдр), октаэдр
Применение преобразований: translate, scale, rotate_x/y/z, rotate_axis, scale_about_center
Визуализация: matplotlib 3D
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math


# ---------------------
# Утилиты для однородных координат
# ---------------------
def to_homogeneous(points):
    """points: (N,3) array -> (N,4) homogeneous"""
    pts = np.asarray(points, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    return np.hstack([pts, ones])


def from_homogeneous(points_h):
    """points_h: (N,4) -> (N,3)"""
    ph = np.asarray(points_h, dtype=float)
    w = ph[:, 3:4]
    # если w == 0 — оставляем как есть (редко в аффинных преобразованиях)
    with np.errstate(divide='ignore', invalid='ignore'):
        coords = ph[:, :3] / w
    return coords


# ---------------------
# Матрицы 4x4 (аффинные)
# ---------------------
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
    """Scale about given point (px,py,pz)."""
    px, py, pz = point
    return mat_translate(px, py, pz) @ mat_scale(sx, sy, sz) @ mat_translate(-px, -py, -pz)


def mat_rotate_x(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[1, 1] = c;
    M[1, 2] = -s
    M[2, 1] = s;
    M[2, 2] = c
    return M


def mat_rotate_y(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0, 0] = c;
    M[0, 2] = s
    M[2, 0] = -s;
    M[2, 2] = c
    return M


def mat_rotate_z(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    M = mat_identity()
    M[0, 0] = c;
    M[0, 1] = -s
    M[1, 0] = s;
    M[1, 1] = c
    return M


def mat_rotate_axis(axis, angle_deg):
    """
    Rodrigues rotation around arbitrary axis passing through origin.
    axis: (3,) vector
    """
    ux, uy, uz = np.asarray(axis, dtype=float)
    norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    if norm == 0:
        return mat_identity()
    ux, uy, uz = ux / norm, uy / norm, uz / norm
    a = math.radians(angle_deg)
    c = math.cos(a);
    s = math.sin(a)
    R = np.array([
        [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s, 0],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s, 0],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c), 0],
        [0, 0, 0, 1]
    ], dtype=float)
    return R


# ---------------------
# НОВЫЕ ФУНКЦИИ ПРЕОБРАЗОВАНИЙ
# ---------------------
def mat_reflection_xy():
    """Отражение относительно плоскости XY (z -> -z)"""
    M = mat_identity()
    M[2, 2] = -1
    return M


def mat_reflection_xz():
    """Отражение относительно плоскости XZ (y -> -y)"""
    M = mat_identity()
    M[1, 1] = -1
    return M


def mat_reflection_yz():
    """Отражение относительно плоскости YZ (x -> -x)"""
    M = mat_identity()
    M[0, 0] = -1
    return M


def mat_reflection_plane(plane):
    """
    Отражение относительно выбранной координатной плоскости.
    plane: 'xy', 'xz', или 'yz'
    """
    if plane == 'xy':
        return mat_reflection_xy()
    elif plane == 'xz':
        return mat_reflection_xz()
    elif plane == 'yz':
        return mat_reflection_yz()
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")


def mat_scale_about_center(sx, sy, sz, polyhedron):
    """Масштабирование многогранника относительно своего центра"""
    center = polyhedron.centroid()
    return mat_scale_about_point(sx, sy, sz, center)


# ---------------------
# НОВЫЕ ФУНКЦИИ ДЛЯ ВРАЩЕНИЙ И ПРОЕКЦИЙ
# ---------------------
def mat_rotate_axis_through_center_parallel(axis_direction, angle_deg, center):
    """
    Вращение вокруг оси, проходящей через центр многогранника 
    и параллельной координатной оси.
    axis_direction: 'x', 'y', или 'z'
    center: точка центра (3,)
    angle_deg: угол в градусах
    """
    # Смещаем в начало координат -> вращаем -> возвращаем обратно
    M_translate_to_origin = mat_translate(-center[0], -center[1], -center[2])
    M_translate_back = mat_translate(center[0], center[1], center[2])
    
    if axis_direction == 'x':
        M_rotate = mat_rotate_x(angle_deg)
    elif axis_direction == 'y':
        M_rotate = mat_rotate_y(angle_deg)
    elif axis_direction == 'z':
        M_rotate = mat_rotate_z(angle_deg)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return M_translate_back @ M_rotate @ M_translate_to_origin


def mat_rotate_arbitrary_axis(point1, point2, angle_deg):
    """
    Поворот вокруг произвольной прямой, заданной двумя точками.
    point1, point2: точки (3,) определяющие прямую
    angle_deg: угол в градусах
    """
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    
    # Вектор направления оси
    axis_vector = p2 - p1
    
    # Перемещаем ось в начало координат
    M_translate_to_origin = mat_translate(-p1[0], -p1[1], -p1[2])
    M_translate_back = mat_translate(p1[0], p1[1], p1[2])
    
    # Вращаем вокруг оси
    M_rotate = mat_rotate_axis(axis_vector, angle_deg)
    
    return M_translate_back @ M_rotate @ M_translate_to_origin


def mat_perspective_projection(d=2.0):
    """Перспективная проекция с расстоянием d от наблюдателя"""
    M = np.eye(4, dtype=float)
    M[3, 2] = -1/d  # Перспективное искажение
    return M


def mat_axonometric_projection():
    """Аксонометрическая проекция (ортографическая)"""
    M = np.eye(4, dtype=float)
    M[2, 2] = 0  # Обнуляем z-координату
    return M


def mat_isometric_projection():
    """Изометрическая проекция"""
    # Поворот на 45° вокруг Y, затем на 35.264° вокруг X
    M_rotY = mat_rotate_y(45)
    M_rotX = mat_rotate_x(35.264)
    M_ortho = mat_axonometric_projection()
    return M_ortho @ M_rotX @ M_rotY


# ---------------------
# Классы геометрии
# ---------------------
class Point3D:
    def __init__(self, x, y, z):
        self.coords = np.array([float(x), float(y), float(z)], dtype=float)

    def as_array(self):
        return self.coords.copy()

    def transform(self, M):
        p_h = to_homogeneous(self.coords.reshape(1, 3))  # (1,4)
        p2 = (p_h @ M.T)
        return Point3D(*from_homogeneous(p2)[0])

    def __repr__(self):
        return f"Point3D({self.coords[0]:.3f}, {self.coords[1]:.3f}, {self.coords[2]:.3f})"


class Polygon3D:
    def __init__(self, vertices):
        """
        vertices: list of Point3D or Nx3 arrays
        """
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
        """polygons: list of Polygon3D"""
        self.polygons = polygons if polygons is not None else []

    def vertices_array(self):
        """Возвращает все вершины как массив (может дублировать одинаковые вершины)."""
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
        """Возвращает новый Polyhedron, трансформированный матрицей M."""
        return Polyhedron([poly.transform(M) for poly in self.polygons])

    def apply_inplace(self, M):
        """Модифицирует текущий объект (in-place)."""
        self.polygons = [poly.transform(M) for poly in self.polygons]

    # НОВЫЕ МЕТОДЫ ПРЕОБРАЗОВАНИЙ
    def reflect_plane(self, plane):
        """Отражение относительно координатной плоскости"""
        M = mat_reflection_plane(plane)
        return self.transform(M)

    def scale_about_center(self, sx, sy, sz):
        """Масштабирование относительно центра многогранника"""
        M = mat_scale_about_center(sx, sy, sz, self)
        return self.transform(M)
    
    # НОВЫЕ МЕТОДЫ ДЛЯ ВРАЩЕНИЙ И ПРОЕКЦИЙ
    def rotate_axis_through_center_parallel(self, axis, angle_deg):
        """Вращение вокруг оси через центр, параллельной координатной"""
        center = self.centroid()
        M = mat_rotate_axis_through_center_parallel(axis, angle_deg, center)
        return self.transform(M)
    
    def rotate_arbitrary_axis(self, point1, point2, angle_deg):
        """Вращение вокруг произвольной прямой"""
        M = mat_rotate_arbitrary_axis(point1, point2, angle_deg)
        return self.transform(M)
    
    def apply_projection(self, projection_type='axonometric', **kwargs):
        """Применение проекции"""
        if projection_type == 'perspective':
            d = kwargs.get('d', 2.0)
            M = mat_perspective_projection(d)
        elif projection_type == 'axonometric':
            M = mat_axonometric_projection()
        elif projection_type == 'isometric':
            M = mat_isometric_projection()
        else:
            raise ValueError("Unknown projection type")
        return self.transform(M)

    # --- Визуализация ---
    def plot(self, ax=None, face_color=(0.7, 0.8, 1.0), edge_color='k', alpha=0.9, linewidth=1):
        """Построить многогранник в заданной 3D-оси matplotlib."""
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False
        poly_verts = [poly.to_array() for poly in self.polygons]
        # Poly3DCollection expects list of (N,3) arrays
        coll = Poly3DCollection(poly_verts, facecolors=face_color, edgecolors=edge_color,
                                linewidths=linewidth, alpha=alpha)
        ax.add_collection3d(coll)
        # autoscale
        all_pts = self.vertices_array()
        if all_pts.size == 0:
            return ax
        xmin, ymin, zmin = all_pts.min(axis=0)
        xmax, ymax, zmax = all_pts.max(axis=0)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        # center
        cx = 0.5 * (xmax + xmin)
        cy = 0.5 * (ymax + ymin)
        cz = 0.5 * (zmax + zmin)
        # set limits
        half = max_range / 2 + 1e-6
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_box_aspect([1, 1, 1])
        if created_fig:
            plt.show()
        return ax


# ---------------------
# Построение правильных многогранников (центр в 0, масштаб 1)
# ---------------------
def make_tetrahedron():
    # правильный тетраэдр: 4 вершины
    # координаты для центрированного тетраэдра (вписаны в сферу радиуса 1)
    a = 1.0
    verts = [
        (a, a, a),
        (a, -a, -a),
        (-a, a, -a),
        (-a, -a, a),
    ]
    faces = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]
    polys = []
    for f in faces:
        polys.append(Polygon3D([Point3D(*verts[i]) for i in f]))
    return Polyhedron(polys)


def make_cube():
    # куб со стороной 2, центр 0
    v = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces_idx = [
        [0, 1, 2, 3],  # нижняя
        [4, 7, 6, 5],  # верхняя
        [0, 4, 5, 1],  # передняя
        [1, 5, 6, 2],  # правая
        [2, 6, 7, 3],  # задняя
        [3, 7, 4, 0],  # левая
    ]
    polys = []
    for f in faces_idx:
        polys.append(Polygon3D([Point3D(*v[i]) for i in f]))
    return Polyhedron(polys)


def make_octahedron():
    # октаэдр: вершины на осях
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces_idx = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]
    ]
    polys = []
    for f in faces_idx:
        polys.append(Polygon3D([Point3D(*verts[i]) for i in f]))
    return Polyhedron(polys)


# ---------------------
# ДЕМОНСТРАЦИЯ НОВЫХ ФУНКЦИЙ
# ---------------------
def demo_new_transformations():
    """Демонстрация новых преобразований"""

    # Создаём куб и размещаем его в первой координатной четверти (x>0, y>0, z>0)
    cube = make_cube()
    # Смещаем куб в первую октанту (x>0, y>0, z>0)
    initial_cube = cube.transform(mat_translate(1.5, 1.5, 1.5))

    # Применяем отражения
    reflected_xy = initial_cube.reflect_plane('xy')  # Отражение относительно XY (z -> -z)
    reflected_xz = initial_cube.reflect_plane('xz')  # Отражение относительно XZ (y -> -y)
    reflected_yz = initial_cube.reflect_plane('yz')  # Отражение относительно YZ (x -> -x)

    # Масштабирование относительно центра - делаем более заметное преобразование
    scaled_center = initial_cube.scale_about_center(1.8, 0.6, 1.4)  # Более выраженное масштабирование

    # Визуализация результатов
    fig = plt.figure(figsize=(20, 12))

    def setup_coordinate_system_small(ax):
        """Настройка компактной координатной системы с плоскостями"""
        # Уменьшенные координатные плоскости
        size = 3
        x_plane = np.array([[-size, -size, 0], [size, -size, 0], [size, size, 0], [-size, size, 0]])
        y_plane = np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]])
        z_plane = np.array([[0, -size, -size], [0, size, -size], [0, size, size], [0, -size, size]])

        ax.add_collection3d(Poly3DCollection([x_plane], alpha=0.15, facecolor='red', label='XY plane'))
        ax.add_collection3d(Poly3DCollection([y_plane], alpha=0.15, facecolor='green', label='XZ plane'))
        ax.add_collection3d(Poly3DCollection([z_plane], alpha=0.15, facecolor='blue', label='YZ plane'))

        # Координатные оси
        axis_length = 4
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='X')
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='Y')
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2, label='Z')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        # Включаем интерактивное управление камерой
        ax.mouse_init()

    # 1. Исходный куб в первой октанте
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    setup_coordinate_system_small(ax1)
    initial_cube.plot(ax=ax1, face_color=(0.2, 0.8, 0.2), alpha=0.8)
    ax1.set_title("Исходный куб\n(первая октанта: x>0, y>0, z>0)")

    # 2. Отражение относительно XY
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    setup_coordinate_system_small(ax2)
    initial_cube.plot(ax=ax2, face_color=(0.2, 0.8, 0.2), alpha=0.3)  # Полупрозрачный исходный
    reflected_xy.plot(ax=ax2, face_color=(0.8, 0.2, 0.2), alpha=0.8)  # Отраженный
    ax2.set_title("Отражение XY\n(z → -z)\nПереход в октанту x>0, y>0, z<0")

    # 3. Отражение относительно XZ
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    setup_coordinate_system_small(ax3)
    initial_cube.plot(ax=ax3, face_color=(0.2, 0.8, 0.2), alpha=0.3)  # Полупрозрачный исходный
    reflected_xz.plot(ax=ax3, face_color=(0.2, 0.2, 0.8), alpha=0.8)  # Отраженный
    ax3.set_title("Отражение XZ\n(y → -y)\nПереход в октанту x>0, y<0, z>0")

    # 4. Отражение относительно YZ
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    setup_coordinate_system_small(ax4)
    initial_cube.plot(ax=ax4, face_color=(0.2, 0.8, 0.2), alpha=0.3)  # Полупрозрачный исходный
    reflected_yz.plot(ax=ax4, face_color=(0.8, 0.8, 0.2), alpha=0.8)  # Отраженный
    ax4.set_title("Отражение YZ\n(x → -x)\nПереход в октанту x<0, y>0, z>0")

    # 5. Масштабирование относительно центра - отдельная демонстрация
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    # Для масштабирования не рисуем координатные плоскости, чтобы лучше видеть форму
    initial_cube.plot(ax=ax5, face_color=(0.2, 0.8, 0.2), alpha=0.6, edge_color='darkgreen')
    scaled_center.plot(ax=ax5, face_color=(0.8, 0.2, 0.8), alpha=0.8, edge_color='purple')

    # Добавляем подписи для ясности
    center = initial_cube.centroid()
    ax5.scatter(*center, color='black', s=50, label='Центр')
    ax5.legend()

    ax5.set_xlim(-3, 5)
    ax5.set_ylim(-3, 5)
    ax5.set_zlim(-3, 5)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_box_aspect([1, 1, 1])
    ax5.mouse_init()
    ax5.set_title("Масштабирование относительно центра\nКуб → Параллелепипед\n(1.8, 0.6, 1.4)")

    # 6. Все отражения вместе
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    setup_coordinate_system_small(ax6)
    initial_cube.plot(ax=ax6, face_color=(0.2, 0.8, 0.2), alpha=0.8, edge_color='darkgreen')
    reflected_xy.plot(ax=ax6, face_color=(0.8, 0.2, 0.2), alpha=0.6, edge_color='darkred')
    reflected_xz.plot(ax=ax6, face_color=(0.2, 0.2, 0.8), alpha=0.6, edge_color='darkblue')
    reflected_yz.plot(ax=ax6, face_color=(0.8, 0.8, 0.2), alpha=0.6, edge_color='goldenrod')
    ax6.set_title("Все отражения вместе\n(разные октанты)")

    plt.tight_layout()
    plt.suptitle(
        "Демонстрация отражений и масштабирования\nКуб перемещается между координатными октантами\n\n",
        y=1.02, fontsize=14)
    plt.show()


# ---------------------
# ДЕМОНСТРАЦИЯ НОВЫХ ВРАЩЕНИЙ И ПРОЕКЦИЙ
# ---------------------
def demo_rotations_and_projections():
    """Демонстрация новых вращений и проекций"""
    
    # Создаем октаэдр для демонстрации
    octa = make_octahedron()
    
    # Применяем смещение для лучшей видимости
    octa = octa.transform(mat_translate(0, 0, 0.5))
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Вращение вокруг оси через центр, параллельной X
    octa_rot_x = octa.rotate_axis_through_center_parallel('x', 45)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    octa.plot(ax=ax1, face_color=(0.8, 0.6, 0.6), alpha=0.7)
    octa_rot_x.plot(ax=ax1, face_color=(0.6, 0.8, 0.6), alpha=0.7)
    ax1.set_title("Вращение вокруг оси через центр\nпараллельной X (45°)")
    ax1.legend(["Исходный", "Повернутый"])
    
    # 2. Вращение вокруг оси через центр, параллельной Y
    octa_rot_y = octa.rotate_axis_through_center_parallel('y', 30)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    octa.plot(ax=ax2, face_color=(0.8, 0.6, 0.6), alpha=0.7)
    octa_rot_y.plot(ax=ax2, face_color=(0.6, 0.6, 0.8), alpha=0.7)
    ax2.set_title("Вращение вокруг оси через центр\nпараллельной Y (30°)")
    ax2.legend(["Исходный", "Повернутый"])
    
    # 3. Вращение вокруг произвольной оси
    point1 = [-2, -2, -2]  # Первая точка оси
    point2 = [2, 2, 2]     # Вторая точка оси
    octa_rot_arb = octa.rotate_arbitrary_axis(point1, point2, 60)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    octa.plot(ax=ax3, face_color=(0.8, 0.6, 0.6), alpha=0.7)
    octa_rot_arb.plot(ax=ax3, face_color=(0.8, 0.8, 0.4), alpha=0.7)
    
    # Рисуем ось вращения
    ax3.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 
             'r-', linewidth=3, label='Ось вращения')
    ax3.set_title("Вращение вокруг произвольной оси\n(60°)")
    ax3.legend(["Исходный", "Повернутый", "Ось вращения"])
    
    # 4. Аксонометрическая проекция
    octa_axono = octa.apply_projection('axonometric')
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    octa_axono.plot(ax=ax4, face_color=(0.7, 0.8, 0.9), alpha=0.8)
    ax4.set_title("Аксонометрическая проекция\n(ортографическая)")
    
    # 5. Перспективная проекция (d=3)
    octa_persp = octa.apply_projection('perspective', d=3)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    octa_persp.plot(ax=ax5, face_color=(0.9, 0.7, 0.8), alpha=0.8)
    ax5.set_title("Перспективная проекция\n(d=3)")
    
    # 6. Изометрическая проекция
    octa_iso = octa.apply_projection('isometric')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    octa_iso.plot(ax=ax6, face_color=(0.8, 0.9, 0.7), alpha=0.8)
    ax6.set_title("Изометрическая проекция")
    
    plt.tight_layout()
    plt.suptitle("Демонстрация новых вращений и проекций\n", y=1.02, fontsize=14)
    plt.show()


# ---------------------
# ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ ПРОЕКЦИЙ
# ---------------------
def demo_projection_comparison():
    """Сравнение разных проекций на одном многограннике"""
    
    # Создаем тетраэдр и применяем к нему различные проекции
    tet = make_tetrahedron()
    tet = tet.transform(mat_scale(1.5, 1.5, 1.5))  # Увеличиваем для наглядности
    
    projections = [
        ('Без проекции', None),
        ('Аксонометрическая', 'axonometric'),
        ('Перспективная (d=2)', lambda: tet.apply_projection('perspective', d=2)),
        ('Перспективная (d=4)', lambda: tet.apply_projection('perspective', d=4)),
        ('Изометрическая', 'isometric')
    ]
    
    fig = plt.figure(figsize=(20, 8))
    
    for i, (title, proj_type) in enumerate(projections):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        if proj_type is None:
            # Без проекции
            tet.plot(ax=ax, face_color=(0.8, 0.7, 0.9), alpha=0.8)
        elif callable(proj_type):
            # Для перспективных проекций с разными параметрами
            tet_proj = proj_type()
            tet_proj.plot(ax=ax, face_color=(0.7, 0.9, 0.8), alpha=0.8)
        else:
            # Стандартные проекции
            tet_proj = tet.apply_projection(proj_type)
            tet_proj.plot(ax=ax, face_color=(0.9, 0.8, 0.7), alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
    
    plt.tight_layout()
    plt.suptitle("Сравнение различных проекций тетраэдра\n", y=1.02, fontsize=14)
    plt.show()


# ---------------------
# Примеры использования
# ---------------------
def demo():
    # Создаём фигуры
    tet = make_tetrahedron()
    cube = make_cube()
    octa = make_octahedron()

    # Отрисуем исходные фигуры рядом
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
    tet.plot(ax=axs[0], face_color=(1.0, 0.6, 0.6));
    axs[0].set_title("Tetrahedron")
    cube.plot(ax=axs[1], face_color=(0.6, 1.0, 0.6));
    axs[1].set_title("Cube")
    octa.plot(ax=axs[2], face_color=(0.6, 0.6, 1.0));
    axs[2].set_title("Octahedron")
    plt.suptitle("Исходные правильные многогранники")
    plt.show()

    # Применим аффинные преобразования к кубу: масштаб относительно центра, поворот, смещение
    # Пример параметров:
    tx, ty, tz = 2.0, 0.5, -1.0  # смещение
    sx, sy, sz = 0.8, 1.5, 0.6  # масштаб
    angle = 40  # градусы поворота вокруг произвольной оси

    # создаём матрицу: сначала масштаб относительно центра куба, затем поворот вокруг оси (1,1,0), затем translate
    center = cube.centroid()
    M_scale_center = mat_scale_about_point(sx, sy, sz, center)
    M_rotate = mat_rotate_axis((1, 1, 0), angle)
    M_translate = mat_translate(tx, ty, tz)
    M = M_translate @ M_rotate @ M_scale_center

    transformed_cube = cube.transform(M)

    # Покажем исходный и трансформированный кубы
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    cube.plot(ax=ax1, face_color=(0.7, 0.9, 0.7));
    ax1.set_title("Исходный куб")
    ax2 = fig.add_subplot(122, projection='3d')
    transformed_cube.plot(ax=ax2, face_color=(0.9, 0.7, 0.7));
    ax2.set_title(
        f"Куб после: scale({sx},{sy},{sz}) about center, rotate {angle}° axis(1,1,0), translate({tx},{ty},{tz})")
    plt.show()

    # Демонстрация последовательных преобразований на тетраэдре
    M1 = mat_rotate_z(30)
    M2 = mat_translate(0.5, 0.0, 0.2)
    M3 = mat_scale(1.2, 1.2, 1.2)
    M_total = M3 @ M2 @ M1
    transformed_tet = tet.transform(M_total)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    tet.plot(ax=ax, face_color=(1, 0.8, 0.6));
    ax.set_title("Исходный тетраэдр")
    ax2 = fig.add_subplot(122, projection='3d')
    transformed_tet.plot(ax=ax2, face_color=(0.6, 0.8, 1.0));
    ax2.set_title("Тетраэдр после последовательных преобразований")
    plt.show()


if __name__ == "__main__":
    demo()
    demo_new_transformations()  # Запуск демонстрации новых функций
    demo_rotations_and_projections()  # Демонстрация вращений и проекций
    demo_projection_comparison()  # Сравнение проекций