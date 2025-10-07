import matplotlib.pyplot as plt
import math


def bresenham(x0, y0, x1, y1):
    """Целочисленный алгоритм Брезенхэма"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def plot_bresenham(ax, x0, y0, x1, y1, color="black"):
    points = bresenham(x0, y0, x1, y1)
    xs, ys = zip(*points)
    ax.scatter(xs, ys, c=color, s=20)


def fpart(x):
    return x - math.floor(x)


def rfpart(x):
    return 1 - fpart(x)


def wu_line(x0, y0, x1, y1, ax, color="red"):
    """Алгоритм Ву с антиалиасингом"""
    def plot(x, y, c):
        ax.scatter([x], [y], c=color, alpha=c, s=40)

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1

    # первая точка
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = int(xend)
    ypxl1 = int(yend)

    if steep:
        plot(ypxl1, xpxl1, rfpart(yend) * xgap)
        plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap)
    else:
        plot(xpxl1, ypxl1, rfpart(yend) * xgap)
        plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap)

    intery = yend + gradient

    # вторая точка
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = int(xend)
    ypxl2 = int(yend)

    # промежуток
    for x in range(xpxl1 + 1, xpxl2):
        if steep:
            plot(int(intery), x, rfpart(intery))
            plot(int(intery) + 1, x, fpart(intery))
        else:
            plot(x, int(intery), rfpart(intery))
            plot(x, int(intery) + 1, fpart(intery))
        intery += gradient

    if steep:
        plot(ypxl2, xpxl2, rfpart(yend) * xgap)
        plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap)
    else:
        plot(xpxl2, ypxl2, rfpart(yend) * xgap)
        plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap)


# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Брезенхем")
plot_bresenham(ax1, 2, 2, 15, 10, "blue")
plot_bresenham(ax1, 2, 2, 15, 2, "green")
plot_bresenham(ax1, 2, 2, 10, 15, "red")

ax2.set_title("Ву")
wu_line(2, 2, 15, 10, ax2, "blue")
wu_line(2, 2, 15, 2, ax2, "green")
wu_line(2, 2, 10, 15, ax2, "red")

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, which="both")

plt.show()