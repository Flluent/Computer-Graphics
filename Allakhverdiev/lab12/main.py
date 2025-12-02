import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

# Вершины правильного тетраэдра
vertices = [
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1]
]

# Ребра (по два индекса вершин)
edges = [
    (0, 1), (0,2), (0,3),
    (1,2), (1,3), (2,3)
]

# Грани (по три индекса вершин) — нужны для заливки
faces = [
    (0,1,2),
    (0,1,3),
    (0,2,3),
    (1,2,3)
]

# Цвета для каждой вершины (RGBA)
colors = [
    (1.0, 0.0, 0.0, 1.0),  # красный
    (0.0, 1.0, 0.0, 1.0),  # зелёный
    (0.0, 0.0, 1.0, 1.0),  # синий
    (1.0, 1.0, 0.0, 1.0)   # жёлтый
]

# Позиция тетраэдра
position = [0.0, 0.0, -6.0]

# Скорость движения
speed = 0.2

def draw_tetrahedron():
    glBegin(GL_TRIANGLES)
    for i, face in enumerate(faces):
        for vertex_idx in face:
            glColor4fv(colors[vertex_idx])
            glVertex3fv(vertices[vertex_idx])
    glEnd()

    # Рисуем рёбра чёрным цветом для лучшей видимости
    glColor4f(0, 0, 0, 1)
    glLineWidth(2)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex_idx in edge:
            glVertex3fv(vertices[vertex_idx])
    glEnd()

def main():
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Градиентный тетраэдр — управление стрелками и WASD")

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(position[0], position[1], position[2])

    # Хороший начальный поворот, чтобы сразу было видно объём
    glRotatef(30, 1, 0, 0)
    glRotatef(-30, 0, 1, 0)

    glEnable(GL_DEPTH_TEST)
    # Включаем плавное сглаживание цветов (гуродово затенение)
    glShadeModel(GL_SMOOTH)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Управление клавиатурой
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            glTranslatef(-speed, 0, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            glTranslatef(speed, 0, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            glTranslatef(0, speed, 0)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            glTranslatef(0, -speed, 0)
        if keys[pygame.K_PAGEUP]:   # ближе
            glTranslatef(0, 0, speed)
        if keys[pygame.K_PAGEDOWN]: # дальше
            glTranslatef(0, 0, -speed)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_tetrahedron()
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()