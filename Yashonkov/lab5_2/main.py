import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Slider, Button


def midpoint_displacement(points, roughness, iterations):
    """
    Генерация горного профиля методом midpoint displacement.
    """
    for i in range(iterations):
        new_points = []
        for j in range(len(points) - 1):
            x1, y1 = points[j]
            x2, y2 = points[j + 1]

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            displacement = (random.random() - 0.5) * (y2 - y1 + 1) * roughness
            new_y = mid_y + displacement

            new_points.append((x1, y1))
            new_points.append((mid_x, new_y))
        new_points.append(points[-1])
        points = new_points

    return points


def generate_mountains(roughness, iterations, ax):
    """
    Перестраивает и отображает ландшафт.
    """
    points = [(0.0, 0.0), (1.0, 0.0)]
    result = midpoint_displacement(points, roughness, iterations)

    ax.clear()
    ax.plot([p[0] for p in result], [p[1] for p in result], color="brown")
    ax.fill_between([p[0] for p in result], [p[1] for p in result], -1, color="sandybrown")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)
    ax.set_title(f"Midpoint Displacement — roughness={roughness:.2f}, iterations={iterations}")
    plt.draw()


# --- Создание интерфейса ---
fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(left=0.1, bottom=0.3)

# Начальные параметры
init_roughness = 0.5
init_iterations = 6
generate_mountains(init_roughness, init_iterations, ax)

# --- Слайдеры ---
axcolor = 'lightgoldenrodyellow'
ax_rough = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_iter = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

slider_rough = Slider(ax_rough, 'Roughness', 0.1, 1.0, valinit=init_roughness, valstep=0.05)
slider_iter = Slider(ax_iter, 'Iterations', 1, 10, valinit=init_iterations, valstep=1)

# --- Кнопка обновления ---
ax_button = plt.axes([0.45, 0.02, 0.1, 0.04])
button = Button(ax_button, 'Обновить', color='lightgray', hovercolor='0.8')


def update(event=None):
    rough = slider_rough.val
    iters = int(slider_iter.val)
    generate_mountains(rough, iters, ax)


button.on_clicked(update)

plt.show()
