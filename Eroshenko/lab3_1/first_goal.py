import pygame
import sys
import os
from pygame.locals import *

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Алгоритмы заливки и выделения границ")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

mode = "draw"
pattern_image = None
boundary_points = []

def scanline_fill_recursive(surface, x, y, fill_color, target_color=None):
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return

    if target_color is None:
        target_color = surface.get_at((x, y))

    current_color = surface.get_at((x, y))
    if current_color == fill_color or current_color != target_color:
        return

    left = x
    while left > 0 and surface.get_at((left - 1, y)) == target_color:
        left -= 1

    right = x
    while right < WIDTH - 1 and surface.get_at((right + 1, y)) == target_color:
        right += 1

    for i in range(left, right + 1):
        surface.set_at((i, y), fill_color)

    for i in range(left, right + 1):
        if y > 0 and surface.get_at((i, y - 1)) == target_color:
            scanline_fill_recursive(surface, i, y - 1, fill_color, target_color)
        if y < HEIGHT - 1 and surface.get_at((i, y + 1)) == target_color:
            scanline_fill_recursive(surface, i, y + 1, fill_color, target_color)

def scanline_pattern_fill_recursive(surface, x, y, pattern_img, target_color=None):
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return

    if target_color is None:
        target_color = surface.get_at((x, y))

    current_color = surface.get_at((x, y))
    if current_color != target_color:
        return

    pattern_width = pattern_img.get_width()
    pattern_height = pattern_img.get_height()

    left = x
    while left > 0 and surface.get_at((left - 1, y)) == target_color:
        left -= 1

    right = x
    while right < WIDTH - 1 and surface.get_at((right + 1, y)) == target_color:
        right += 1

    for i in range(left, right + 1):
        pattern_x = i % pattern_width
        pattern_y = y % pattern_height
        pattern_color = pattern_img.get_at((pattern_x, pattern_y))
        surface.set_at((i, y), pattern_color)

    for i in range(left, right + 1):
        if y > 0 and surface.get_at((i, y - 1)) == target_color:
            scanline_pattern_fill_recursive(surface, i, y - 1, pattern_img, target_color)
        if y < HEIGHT - 1 and surface.get_at((i, y + 1)) == target_color:
            scanline_pattern_fill_recursive(surface, i, y + 1, pattern_img, target_color)

def find_boundary(surface, start_x, start_y, boundary_color):
    boundary_points = []
    visited = set()

    directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                  (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def is_boundary_point(x, y):
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False

        if surface.get_at((x, y)) != boundary_color:
            return False

        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT:
                return True
            if surface.get_at((nx, ny)) != boundary_color:
                return True

        return False

    if not is_boundary_point(start_x, start_y):
        found = False
        for radius in range(1, 100):
            for dx in range(-radius, radius + 1):
                for dy in [-radius, radius]:
                    for point in [(start_x + dx, start_y + dy)]:
                        x, y = point
                        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                            if is_boundary_point(x, y):
                                start_x, start_y = x, y
                                found = True
                                break
                if found:
                    break
            if found:
                break

    if not is_boundary_point(start_x, start_y):
        return []

    current_x, current_y = start_x, start_y
    first_point = (start_x, start_y)
    first_dir = 0

    while True:
        if (current_x, current_y) not in visited:
            boundary_points.append((current_x, current_y))
            visited.add((current_x, current_y))

        found_next = False
        next_x, next_y = current_x, current_y

        for i in range(8):
            dir_idx = (first_dir + i) % 8
            dx, dy = directions[dir_idx]
            nx, ny = current_x + dx, current_y + dy

            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                if is_boundary_point(nx, ny) and (nx, ny) not in visited:
                    next_x, next_y = nx, ny
                    first_dir = (dir_idx + 5) % 8
                    found_next = True
                    break

        if not found_next:
            break

        if (next_x, next_y) == first_point and len(boundary_points) > 1:
            break

        current_x, current_y = next_x, next_y

        if len(boundary_points) > 2000:
            break

    return boundary_points

def draw_test_shapes(surface):
    surface.fill(BLACK)

    pygame.draw.circle(surface, WHITE, (200, 150), 50, 0)

    pygame.draw.rect(surface, WHITE, (300, 100, 100, 80), 0)

    points = [(500, 100), (450, 180), (550, 180)]
    pygame.draw.polygon(surface, WHITE, points, 0)

def load_pattern_file():
    global pattern_image

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Выберите рисунок для заливки",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    if file_path:
        try:
            pattern_image = pygame.image.load(file_path)
            print(f"Рисунок для заливки загружен: {file_path}")
            print(f"Размер рисунка: {pattern_image.get_width()}x{pattern_image.get_height()}")
        except pygame.error as e:
            print(f"Ошибка загрузки рисунка: {e}")
            pattern_image = None

def draw_interface():
    font = pygame.font.SysFont(None, 24)

    modes = [
        ("Рисование", "draw"),
        ("Заливка цветом", "fill_color"),
        ("Заливка рисунком", "fill_pattern"),
        ("Найти границу", "find_boundary"),
        ("Тест фигуры", "test_shapes"),
        ("Загрузить рисунок", "load_pattern")
    ]

    for i, (text, mode_name) in enumerate(modes):
        color = YELLOW if mode == mode_name else GRAY
        pygame.draw.rect(screen, color, (10 + i * 150, 10, 140, 30))
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (15 + i * 150, 15))

    info_text = f"Режим: {mode}. ЛКМ - действие, ПКМ - очистка"
    if pattern_image:
        info_text += f" | Рисунок: {pattern_image.get_width()}x{pattern_image.get_height()}"

    info_surface = font.render(info_text, True, WHITE)
    screen.blit(info_surface, (10, HEIGHT - 30))

    if boundary_points and mode == "find_boundary":
        points_info = font.render(f"Точек границы: {len(boundary_points)}", True, GREEN)
        screen.blit(points_info, (10, HEIGHT - 60))

def main():
    global mode, pattern_image, boundary_points

    drawing_surface = pygame.Surface((WIDTH, HEIGHT))
    drawing_surface.fill(BLACK)

    drawing = False
    last_pos = None

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == MOUSEBUTTONDOWN:
                x, y = event.pos

                if 10 <= y <= 40:
                    for i, (_, mode_name) in enumerate([
                        ("Рисование", "draw"),
                        ("Заливка цветом", "fill_color"),
                        ("Заливка рисунком", "fill_pattern"),
                        ("Найти границу", "find_boundary"),
                        ("Тест фигуры", "test_shapes"),
                        ("Загрузить рисунок", "load_pattern")
                    ]):
                        if 10 + i * 150 <= x <= 10 + i * 150 + 140:
                            mode = mode_name
                            boundary_points = []
                            if mode == "test_shapes":
                                draw_test_shapes(drawing_surface)
                            elif mode == "load_pattern":
                                load_pattern_file()
                            break

                elif event.button == 1:
                    if mode == "draw":
                        drawing = True
                        last_pos = (x, y)

                    elif mode == "fill_color":
                        scanline_fill_recursive(drawing_surface, x, y, BLUE)

                    elif mode == "fill_pattern":
                        if pattern_image:
                            scanline_pattern_fill_recursive(drawing_surface, x, y, pattern_image)
                        else:
                            print("Сначала загрузите рисунок для заливки!")

                    elif mode == "find_boundary":
                        boundary_color = drawing_surface.get_at((x, y))
                        print(f"Поиск границы цвета: {boundary_color} в точке ({x}, {y})")
                        boundary_points = find_boundary(drawing_surface, x, y, boundary_color)
                        print(f"Найдено точек границы: {len(boundary_points)}")

                elif event.button == 3:
                    drawing_surface.fill(BLACK)
                    boundary_points = []

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1 and mode == "draw":
                    drawing = False
                    last_pos = None

            elif event.type == MOUSEMOTION:
                if drawing and mode == "draw":
                    current_pos = event.pos
                    if last_pos:
                        pygame.draw.line(drawing_surface, WHITE, last_pos, current_pos, 2)
                    last_pos = current_pos

        screen.fill(BLACK)
        screen.blit(drawing_surface, (0, 0))

        if boundary_points and mode == "find_boundary":
            if len(boundary_points) > 1:
                pygame.draw.lines(screen, RED, True, boundary_points, 2)
            for point in boundary_points:
                pygame.draw.circle(screen, GREEN, point, 1)

        draw_interface()
        pygame.display.flip()

if __name__ == "__main__":
    main()