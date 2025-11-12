import numpy as np
import matplotlib.pyplot as plt
import math


# ---------- Загрузка OBJ ----------
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    # Квад → разбиваем на два треугольника
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    return np.array(vertices), np.array(faces)


# ---------- Трансформации ----------
def rotate_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotate_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def affine_transform(vertices, rx=0, ry=0, scale=1.0, translate=(0, 0, 0)):
    R = rotate_x(rx) @ rotate_y(ry)
    transformed = vertices @ R.T * scale
    transformed += np.array(translate)
    return transformed


# ---------- Растеризация с визуализацией глубины ----------
def render_depth_map(vertices, faces, width=600, height=600):
    img = np.zeros((height, width, 3), dtype=float)
    zbuf = np.full((height, width), np.inf)

    # Центрируем и масштабируем для лучшего отображения
    v = vertices.copy()
    center = np.mean(v[:, :2], axis=0)
    scale = min(width, height) * 0.4 / np.max(np.ptp(v[:, :2], axis=0))

    v[:, 0] = (v[:, 0] - center[0]) * scale + width / 2
    v[:, 1] = (v[:, 1] - center[1]) * scale + height / 2

    # Нормализуем Z для цветового диапазона
    z_min, z_max = np.min(v[:, 2]), np.max(v[:, 2])
    z_range = z_max - z_min

    for face in faces:
        pts = v[face]
        x0, y0, z0 = pts[0]
        x1, y1, z1 = pts[1]
        x2, y2, z2 = pts[2]

        minx = max(0, int(min(x0, x1, x2)))
        maxx = min(width - 1, int(max(x0, x1, x2)))
        miny = max(0, int(min(y0, y1, y2)))
        maxy = min(height - 1, int(max(y0, y1, y2)))

        if minx >= maxx or miny >= maxy:
            continue

        # Back-face culling (удаление нелицевых граней)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.linalg.norm(normal) > 1e-6:
            normal = normal / np.linalg.norm(normal)
            if normal[2] > 0:  # Если грань смотрит от камеры
                continue

        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < 1e-6:
                    continue

                w1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
                w2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
                w3 = 1 - w1 - w2

                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    z = w1 * z0 + w2 * z1 + w3 * z2

                    if z < zbuf[y, x]:
                        zbuf[y, x] = z

                        # Преобразуем глубину в оттенок серого
                        # Ближе = светлее, дальше = темнее
                        if z_range > 1e-6:
                            depth_normalized = (z - z_min) / z_range
                            # Инвертируем: ближние точки светлее
                            gray_value = 1.0 - depth_normalized - 0.1
                        else:
                            gray_value = 0.7

                        # Ограничиваем диапазон и применяем
                        gray_value = max(0.1, min(0.9, gray_value))
                        img[y, x] = [gray_value, gray_value, gray_value]

    return np.flipud(img)


# ---------- Основная часть ----------
vertices, faces = load_obj("cube.obj")

# Применяем вращение для лучшего обзора глубины
vertices = affine_transform(
    vertices,
    rx=math.radians(25),
    ry=math.radians(35),
    scale=1.5
)

img = render_depth_map(vertices, faces)

# Создаем красивый вывод
plt.figure(figsize=(12, 5))

# Основное изображение
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Depth Map Visualization\n(Lighter = Closer, Darker = Farther)", fontsize=12, pad=10)

# Цветовая шкала для справки
plt.subplot(1, 2, 2)
gradient = np.linspace(0.9, 0.1, 100).reshape(100, 1)
gradient = np.repeat(gradient, 50, axis=1)
plt.imshow(gradient, cmap='gray', aspect='auto')
plt.axis("off")
plt.title("Depth Scale", fontsize=12, pad=10)
plt.text(25, 10, "CLOSE", fontsize=10, weight='bold', ha='center', color='white')
plt.text(25, 90, "FAR", fontsize=10, weight='bold', ha='center', color='black')

plt.tight_layout()
plt.show()

# Дополнительно: выводим информацию о глубине
z_values = vertices[:, 2]
print(f"Depth range: {np.min(z_values):.3f} to {np.max(z_values):.3f}")
print(f"Front faces are lighter, back faces are darker")