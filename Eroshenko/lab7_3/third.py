import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    # Задание 3: Построение поверхности z = f(x, y)
    # =====================================================
    @staticmethod
    def create_surface(f, x_range, y_range, n_x, n_y):
        x0, x1 = x_range
        y0, y1 = y_range

        x_vals = np.linspace(x0, x1, n_x + 1)
        y_vals = np.linspace(y0, y1, n_y + 1)

        points = []
        for i, x in enumerate(x_vals):
            row_points = []
            for j, y in enumerate(y_vals):
                try:
                    z = f(x, y)
                    if np.isnan(z) or np.isinf(z):
                        z = 0.0  
                    row_points.append(Point3D(x, y, z))
                except Exception:
                    row_points.append(Point3D(x, y, 0.0))
            points.append(row_points)

  
        polygons = []
        for i in range(n_x):
            for j in range(n_y):
                
                try:
                    poly = Polygon3D([
                        points[i][j],  # левая нижняя
                        points[i + 1][j],  # правая нижняя
                        points[i + 1][j + 1],  # правая верхняя
                        points[i][j + 1]  # левая верхняя
                    ])
                    polygons.append(poly)
                except IndexError:
                    # Пропускаем некорректные полигоны
                    continue

        return Polyhedron(polygons)


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


# =====================================================
# ДОПОЛНИТЕЛЬНЫЙ КОД ДЛЯ ФИГУР ВРАЩЕНИЯ
# =====================================================

def create_revolution_surface(profile_points, axis='y', segments=12):
    """
    Создание фигуры вращения из образующей кривой

    Parameters:
    - profile_points: список точек [(x1, y1, z1), (x2, y2, z2), ...]
    - axis: ось вращения ('x', 'y', 'z')
    - segments: количество разбиений (сегментов вращения)

    Returns:
    - Polyhedron объект
    """

    # Преобразуем точки в массив numpy
    profile = np.array(profile_points, dtype=float)

    # Нормализуем ось вращения
    if axis.lower() == 'x':
        axis_coords = (1, 0, 0)
        # Для вращения вокруг X, профиль задаётся в плоскости YZ
        if profile.shape[1] == 2:
            profile = np.column_stack([np.zeros(len(profile)), profile])
    elif axis.lower() == 'y':
        axis_coords = (0, 1, 0)
        # Для вращения вокруг Y, профиль задаётся в плоскости XZ
        if profile.shape[1] == 2:
            profile = np.column_stack([profile[:, 0], np.zeros(len(profile)), profile[:, 1]])
    elif axis.lower() == 'z':
        axis_coords = (0, 0, 1)
        # Для вращения вокруг Z, профиль задаётся в плоскости XY
        if profile.shape[1] == 2:
            profile = np.column_stack([profile, np.zeros(len(profile))])
    else:
        raise ValueError("Ось должна быть 'x', 'y' или 'z'")

    polygons = []
    angle_step = 360.0 / segments

    # Создаём все вершины вращения
    all_vertices = []
    for i in range(segments + 1):  # +1 для замыкания
        angle = i * angle_step

        # Матрица вращения для текущего угла
        if axis.lower() == 'x':
            M = mat_rotate_x(angle)
        elif axis.lower() == 'y':
            M = mat_rotate_y(angle)
        else:  # 'z'
            M = mat_rotate_z(angle)

        # Применяем вращение к профилю
        profile_h = to_homogeneous(profile)
        rotated_profile = from_homogeneous(profile_h @ M.T)
        all_vertices.append(rotated_profile)

    # Создаём полигоны (грани)
    for i in range(segments):
        current_ring = all_vertices[i]
        next_ring = all_vertices[i + 1]

        for j in range(len(profile) - 1):
            # Создаём четырёхугольную грань между двумя соседними кольцами
            v1 = Point3D(*current_ring[j])
            v2 = Point3D(*current_ring[j + 1])
            v3 = Point3D(*next_ring[j + 1])
            v4 = Point3D(*next_ring[j])

            polygons.append(Polygon3D([v1, v2, v3, v4]))

    return Polyhedron(polygons)


def create_cylinder(radius=1.0, height=2.0, segments=12):
    """Создание цилиндра как фигуры вращения"""
    profile = [
        (radius, -height / 2),
        (radius, height / 2)
    ]
    return create_revolution_surface(profile, axis='y', segments=segments)


def create_cone(radius=1.0, height=2.0, segments=12):
    """Создание конуса как фигуры вращения"""
    profile = [
        (0, -height / 2),
        (radius, -height / 2),
        (0, height / 2)
    ]
    return create_revolution_surface(profile, axis='y', segments=segments)


def create_sphere(radius=1.0, segments=12, slices=8):
    """Создание сферы как фигуры вращения"""
    profile = []
    for i in range(slices + 1):
        theta = i * np.pi / slices - np.pi / 2
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        profile.append((x, y))
    return create_revolution_surface(profile, axis='y', segments=segments)


def create_torus(major_radius=2.0, minor_radius=0.5, segments=12, tube_segments=8):
    """Создание тора как фигуры вращения"""
    profile = []
    for i in range(tube_segments + 1):
        angle = i * 2 * np.pi / tube_segments
        x = major_radius + minor_radius * np.cos(angle)
        y = minor_radius * np.sin(angle)
        profile.append((x, y))
    return create_revolution_surface(profile, axis='y', segments=segments)


# =====================================================
# ДЕМОНСТРАЦИЯ ФИГУР ВРАЩЕНИЯ
# =====================================================

def demo_revolution():
    """Демонстрация создания фигур вращения"""

    # 1. Произвольная фигура вращения
    custom_profile = [
        (0, 0), (1, 0), (1.5, 1), (1, 2), (0.5, 2.5), (0, 2)
    ]

    revolution = create_revolution_surface(
        profile_points=custom_profile,
        axis='y',
        segments=16
    )

    # 2. Стандартные фигуры
    cylinder = create_cylinder(radius=0.5, height=1.5, segments=12)
    cone = create_cone(radius=0.7, height=2.0, segments=12)
    sphere = create_sphere(radius=1.0, segments=16, slices=8)
    torus = create_torus(major_radius=1.5, minor_radius=0.3, segments=16, tube_segments=12)

    # Отображение всех фигур
    fig = plt.figure(figsize=(15, 10))

    # Произвольная фигура вращения
    ax1 = fig.add_subplot(231, projection='3d')
    revolution.plot(ax=ax1, face_color=(0.8, 0.9, 1.0), edge_color='blue')
    ax1.set_title("Произвольная фигура вращения")

    # Цилиндр
    ax2 = fig.add_subplot(232, projection='3d')
    cylinder.transform(mat_translate(0, 0, 0)).plot(ax=ax2, face_color=(1.0, 0.8, 0.8))
    ax2.set_title("Цилиндр")

    # Конус
    ax3 = fig.add_subplot(233, projection='3d')
    cone.transform(mat_translate(0, 0, 0)).plot(ax=ax3, face_color=(0.8, 1.0, 0.8))
    ax3.set_title("Конус")

    # Сфера
    ax4 = fig.add_subplot(234, projection='3d')
    sphere.transform(mat_translate(0, 0, 0)).plot(ax=ax4, face_color=(1.0, 1.0, 0.8))
    ax4.set_title("Сфера")

    # Тор
    ax5 = fig.add_subplot(235, projection='3d')
    torus.transform(mat_translate(0, 0, 0)).plot(ax=ax5, face_color=(1.0, 0.8, 1.0))
    ax5.set_title("Тор")

    # Сохранение произвольной фигуры
    revolution.to_obj("custom_revolution.obj")
    print("Сохранено: custom_revolution.obj")

    # Применение преобразований к одной из фигур
    ax6 = fig.add_subplot(236, projection='3d')
    transformed = revolution.transform(
        mat_rotate_x(45) @ mat_scale(1.2, 0.8, 1.2) @ mat_translate(1, 0, 0)
    )
    transformed.plot(ax=ax6, face_color=(0.9, 0.7, 1.0))
    ax6.set_title("После преобразований")

    plt.tight_layout()
    plt.show()

    # Сохраняем преобразованную фигуру
    transformed.to_obj("transformed_revolution.obj")
    print("Сохранено: transformed_revolution.obj")


def demo_custom_revolution_with_loading():
    """Демонстрация с загрузкой сохранённой модели"""

    # Создаём сложную фигуру вращения
    complex_profile = [
        (0, -2), (0.5, -1.5), (1.2, -1), (1.5, 0),
        (1.2, 1), (0.5, 1.5), (0, 2), (-0.3, 1.5),
        (-0.5, 1), (-0.3, 0), (-0.5, -1), (-0.3, -1.5)
    ]

    # Создаём фигуру вращения
    model = create_revolution_surface(
        profile_points=complex_profile,
        axis='y',
        segments=24
    )

    # Сохраняем
    model.to_obj("complex_revolution.obj")
    print(f"Создано граней: {len(model.polygons)}")
    print("Сохранено: complex_revolution.obj")

    # Загружаем обратно (бонусная часть)
    try:
        loaded_model = Polyhedron.from_obj("complex_revolution.obj")
        print(f"Загружено граней: {len(loaded_model.polygons)}")

        # Применяем преобразования к загруженной модели
        transformed_loaded = loaded_model.transform(
            mat_rotate_y(30) @ mat_scale(0.8, 1.2, 0.8) @ mat_translate(0, 0.5, 0)
        )

        # Отображаем оригинал и преобразованную загруженную модель
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        model.plot(ax=ax1, face_color=(0.7, 0.8, 1.0))
        ax1.set_title("Оригинальная модель")

        ax2 = fig.add_subplot(122, projection='3d')
        transformed_loaded.plot(ax=ax2, face_color=(1.0, 0.7, 0.7))
        ax2.set_title("Загруженная + преобразования")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ошибка при загрузке: {e}")


# =====================================================
# ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ
# =====================================================

def interactive_revolution_demo():
    """Интерактивное создание фигуры вращения"""

    # Простой профиль для демонстрации
    simple_profile = [
        (0, 0), (1, 0), (1.2, 0.5), (0.8, 1), (0.5, 1.2), (0, 1)
    ]

    print("Демонстрация фигур вращения:")
    print("1. Создание фигуры из профиля")
    print("2. Применение аффинных преобразований")
    print("3. Сохранение в файл")
    print("4. Загрузка из файла (бонус)")

    # Создаём базовую фигуру
    base_model = create_revolution_surface(
        profile_points=simple_profile,
        axis='y',
        segments=18
    )

    # Применяем последовательность преобразований
    models = [base_model]
    transformations = [
        ("Исходная", mat_identity()),
        ("Поворот X=45°", mat_rotate_x(45)),
        ("Масштаб (1.5,1,1)", mat_scale(1.5, 1, 1)),
        ("Перенос + вращение", mat_translate(1, 0, 0) @ mat_rotate_y(60)),
        ("Масштаб от центра", mat_scale_about_point(0.7, 1.3, 0.7, base_model.centroid()))
    ]

    colors = [(0.8, 0.9, 1.0), (1.0, 0.9, 0.8), (0.9, 1.0, 0.8),
              (1.0, 0.8, 0.9), (0.8, 1.0, 1.0)]

    # Создаём все преобразованные модели
    for name, M in transformations[1:]:
        models.append(base_model.transform(M))

    # Отображаем все модели
    fig = plt.figure(figsize=(15, 8))
    for i, (model, (name, _), color) in enumerate(zip(models, transformations, colors)):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        model.plot(ax=ax, face_color=color)
        ax.set_title(name)

    plt.tight_layout()
    plt.show()

    # Сохраняем несколько моделей
    for i, model in enumerate(models[:3]):
        filename = f"revolution_model_{i + 1}.obj"
        model.to_obj(filename)
        print(f"Сохранено: {filename}")


#3 
def demo_surface():
   
    print("\n" + "=" * 50)
    print("ЗАДАНИЕ 3: ПОСТРОЕНИЕ ПОВЕРХНОСТИ z = f(x, y)")
    print("=" * 50)

    functions = {
        '1': ('Параболоид: z = x² + y²', lambda x, y: x ** 2 + y ** 2),
        '2': ('Седло: z = x² - y²', lambda x, y: x ** 2 - y ** 2),
        '3': ('Синусоидальная волна: z = sin(x) * cos(y)', lambda x, y: math.sin(x) * math.cos(y)),
        '4': ('Гауссова поверхность: z = exp(-(x² + y²))', lambda x, y: math.exp(-(x ** 2 + y ** 2))),
        '5': ('Поверхность с рябью: z = sin(√(x² + y²))',
              lambda x, y: math.sin(math.sqrt(x ** 2 + y ** 2)) if x ** 2 + y ** 2 > 0 else 0),
        '6': ('Сложная функция: z = sin(x) * cos(y) + 0.1*(x² + y²)',
              lambda x, y: math.sin(x) * math.cos(y) + 0.1 * (x ** 2 + y ** 2)),
        '7': ('Функция сложения', lambda x, y: x + y )
    }

    print("\nВыберите функцию для построения поверхности:")
    for key in functions:
        print(f"{key}. {functions[key][0]}")

    choice = input("\nВведите номер функции (1-6): ")

    if choice not in functions:
        print("Неверный выбор. Используется функция по умолчанию (параболоид).")
        choice = '1'

    func_name, func = functions[choice]

    print(f"\nВыбрана функция: {func_name}")
    print("\nВведите диапазон по оси X:")
    try:
        x_min = float(input("  Минимальное значение X: ") or "-2")
        x_max = float(input("  Максимальное значение X: ") or "2")
    except ValueError:
        print("Некорректный ввод. Используются значения по умолчанию (-2, 2).")
        x_min, x_max = -2, 2

    print("\nВведите диапазон по оси Y:")
    try:
        y_min = float(input("  Минимальное значение Y: ") or "-2")
        y_max = float(input("  Максимальное значение Y: ") or "2")
    except ValueError:
        print("Некорректный ввод. Используются значения по умолчанию (-2, 2).")
        y_min, y_max = -2, 2

    print("\nВведите количество разбиений (шагов):")
    try:
        n = int(input("  Количество разбиений по X и Y: ") or "20")
    except ValueError:
        print("Некорректный ввод. Используется значение по умолчанию (20).")
        n = 20

    print(f"\nПостроение поверхности...")
    print(f"  Диапазон X: [{x_min}, {x_max}]")
    print(f"  Диапазон Y: [{y_min}, {y_max}]")
    print(f"  Количество разбиений: {n} × {n}")

    surface = Polyhedron.create_surface(
        f=func,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        n_x=n,
        n_y=n
    )

    print(f"  Создано граней: {len(surface.polygons)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface.plot(ax=ax, face_color=(0.7, 0.8, 1.0), alpha=0.8)
    ax.set_title(f"Исходная поверхность: {func_name}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    save_original = input("\nСохранить исходную поверхность в файл? (y/n): ")
    if save_original.lower() == 'y':
        filename = input("Введите имя файла (например, surface.obj): ") or "surface.obj"
        surface.to_obj(filename)
        print(f"Сохранено: {filename}")

    apply_transform = input("\nПрименить аффинные преобразования к поверхности? (y/n): ")

    if apply_transform.lower() == 'y':
        print("\nВыберите преобразование:")
        print("1. Поворот вокруг оси X на 30°")
        print("2. Поворот вокруг оси Y на 45°")
        print("3. Масштабирование (1.5, 1, 0.8)")
        print("4. Перенос на (1, 0, 0)")
        print("5. Комбинированное преобразование (поворот + масштаб)")

        transform_choice = input("\nВведите номер преобразования (1-5): ")

        if transform_choice == '1':
            M = mat_rotate_x(30)
            transform_name = "Поворот вокруг X на 30°"
        elif transform_choice == '2':
            M = mat_rotate_y(45)
            transform_name = "Поворот вокруг Y на 45°"
        elif transform_choice == '3':
            M = mat_scale(1.5, 1.0, 0.8)
            transform_name = "Масштабирование (1.5, 1, 0.8)"
        elif transform_choice == '4':
            M = mat_translate(1, 0, 0)
            transform_name = "Перенос на (1, 0, 0)"
        elif transform_choice == '5':
            M = mat_rotate_y(30) @ mat_scale(0.8, 0.8, 1.2) @ mat_translate(0.5, 0, 0)
            transform_name = "Комбинированное преобразование"
        else:
            print("Неверный выбор. Преобразование не применяется.")
            M = mat_identity()
            transform_name = "Без преобразования"

        transformed_surface = surface.transform(M)

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        surface.plot(ax=ax1, face_color=(0.7, 0.8, 1.0), alpha=0.8)
        ax1.set_title("Исходная поверхность")

        ax2 = fig.add_subplot(122, projection='3d')
        transformed_surface.plot(ax=ax2, face_color=(1.0, 0.7, 0.7), alpha=0.8)
        ax2.set_title(f"После преобразования: {transform_name}")

        plt.show()

        save_transformed = input("\nСохранить преобразованную поверхность в файл? (y/n): ")
        if save_transformed.lower() == 'y':
            filename = input("Введите имя файла (например, transformed_surface.obj): ") or "transformed_surface.obj"
            transformed_surface.to_obj(filename)
            print(f"Сохранено: {filename}")

    # Загрузка и отображение сохраненной поверхности
    load_surface = input("\nЗагрузить и отобразить сохраненную поверхность? (y/n): ")

    if load_surface.lower() == 'y':
        filename = input("Введите имя файла для загрузки: ") or "surface.obj"

        try:
            loaded_surface = Polyhedron.from_obj(filename)
            print(f"Загружено граней: {len(loaded_surface.polygons)}")

            # Отображаем загруженную поверхность
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            loaded_surface.plot(ax=ax, face_color=(0.8, 0.9, 0.7), alpha=0.8)
            ax.set_title(f"Загруженная поверхность: {filename}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        except FileNotFoundError:
            print(f"Файл {filename} не найден.")
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")

    print("\n" + "=" * 50)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 50)


# =====================================================
# ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ
# =====================================================

if __name__ == "__main__":
    # Запуск демонстраций одногруппников
    demo_obj_load()

    print("=" * 50)
    print("ДЕМОНСТРАЦИЯ ФИГУР ВРАЩЕНИЯ")
    print("=" * 50)
    demo_revolution()

    print("\n" + "=" * 50)
    print("ДЕМОНСТРАЦИЯ С ЗАГРУЗКОЙ")
    print("=" * 50)
    demo_custom_revolution_with_loading()

    print("\n" + "=" * 50)
    print("ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ")
    print("=" * 50)
    interactive_revolution_demo()

    # Запуск демонстрации задания 3
    demo_surface()
