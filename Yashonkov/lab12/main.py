import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import os

# ==================== ОБЩИЕ ДАННЫЕ ====================
vertices_tetra = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
edges_tetra = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
faces_tetra = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
colors_tetra = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1)]

vertices_cube = [
    [1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
    [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]
]
faces_cube = [(0,1,2,3),(4,5,6,7),(0,3,7,4),(1,2,6,5),(0,1,5,4),(3,2,6,7)]
texcoords = [(0,0),(1,0),(1,1),(0,1)]
colors_cube = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),(1,1,1,1),(0.5,0.5,0.5,1)]

# ==================== ЗАГРУЗКА ТЕКСТУРЫ ====================
def load_texture(path):
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        return None
    surf = pygame.image.load(path).convert_alpha()
    data = pygame.image.tostring(surf, "RGBA", True)
    w, h = surf.get_size()
    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    return tid

# ==================== РИСОВАНИЕ ====================
def draw_tetrahedron():
    glBegin(GL_TRIANGLES)
    for face in faces_tetra:
        for i in face:
            glColor4fv(colors_tetra[i])
            glVertex3fv(vertices_tetra[i])
    glEnd()
    glColor4f(0,0,0,1)
    glLineWidth(2)
    glBegin(GL_LINES)
    for a,b in edges_tetra:
        glVertex3fv(vertices_tetra[a])
        glVertex3fv(vertices_tetra[b])
    glEnd()

def draw_cube_single_texture(tex_id, color_influence):
    if not tex_id:
        return

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    glBegin(GL_QUADS)
    for face in faces_cube:
        for j, vidx in enumerate(face):
            c = colors_cube[vidx]
            r = 1.0*(1-color_influence) + c[0]*color_influence
            g = 1.0*(1-color_influence) + c[1]*color_influence
            b = 1.0*(1-color_influence) + c[2]*color_influence
            glColor4f(r, g, b, 1.0)  # полная непрозрачность
            glTexCoord2f(*texcoords[j])
            glVertex3fv(vertices_cube[vidx])
    glEnd()
    glDisable(GL_TEXTURE_2D)

    # рёбра
    glColor4f(0,0,0,1)
    glLineWidth(2)
    glBegin(GL_LINES)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        glVertex3fv(vertices_cube[a]); glVertex3fv(vertices_cube[b])
    glEnd()

def draw_cube_two_textures(tex1, tex2, blend_factor, color_influence):
    """
    Рисует куб с двумя текстурами:
    - Сначала рисуем базовую текстуру tex1 (полностью непрозрачную, color influence применяется).
    - Затем рисуем tex2 поверх с альфой = blend_factor (0..1).
    Для корректного наложения второй проход выполняется с glDepthFunc(GL_LEQUAL).
    """
    if not tex1 or not tex2:
        # Если одной из текстур нет — fallback: рисуем ту, что есть
        if tex1:
            draw_cube_single_texture(tex1, color_influence)
        elif tex2:
            draw_cube_single_texture(tex2, color_influence)
        return

    # --- PASS 1: базовая текстура (tex1), без блендинга, полностью непрозрачная ---
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex1)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    # Отключаем блендинг для первого прохода (чтобы он записал пиксели в буфер цвета)
    was_blend_enabled = glIsEnabled(GL_BLEND)
    if was_blend_enabled:
        glDisable(GL_BLEND)

    glBegin(GL_QUADS)
    for face in faces_cube:
        for j, vidx in enumerate(face):
            c = colors_cube[vidx]
            r = (1 - color_influence) + c[0] * color_influence
            g = (1 - color_influence) + c[1] * color_influence
            b = (1 - color_influence) + c[2] * color_influence
            glColor4f(r, g, b, 1.0)  # полностью непрозрачный базовый слой
            glTexCoord2f(*texcoords[j])
            glVertex3fv(vertices_cube[vidx])
    glEnd()

    # --- PASS 2: вторая текстура (tex2) — накладывается поверх с alpha = blend_factor ---
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex2)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    # Чтобы второй проход отрисовался поверх первого (фрагменты имеют ту же глубину),
    # разрешаем проход при условии <= (равная глубина тоже пройдет)
    prev_depth_func = glGetIntegerv(GL_DEPTH_FUNC)
    glDepthFunc(GL_LEQUAL)

    glBegin(GL_QUADS)
    for face in faces_cube:
        for j, vidx in enumerate(face):
            c = colors_cube[vidx]
            r = (1 - color_influence) + c[0] * color_influence
            g = (1 - color_influence) + c[1] * color_influence
            b = (1 - color_influence) + c[2] * color_influence
            # Альфа определяет долю второй текстуры в сумме
            glColor4f(r, g, b, blend_factor)
            glTexCoord2f(*texcoords[j])
            glVertex3fv(vertices_cube[vidx])
    glEnd()

    # Восстанавливаем состояние глубины и блендинга
    glDepthFunc(prev_depth_func)
    glDisable(GL_BLEND)
    if was_blend_enabled:
        glEnable(GL_BLEND)

    glDisable(GL_TEXTURE_2D)

    # Рёбра — всегда непрозрачные чёрные
    glColor4f(0, 0, 0, 1)
    glLineWidth(2)
    glBegin(GL_LINES)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        glVertex3fv(vertices_cube[a])
        glVertex3fv(vertices_cube[b])
    glEnd()

# ==================== ОСНОВНОЙ ЦИКЛ ====================
def main():
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 50.0)
    glTranslatef(0, 0, -6)
    glRotatef(25, 1, 0, 0)

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    tex1 = load_texture("texture.png")
    tex2 = load_texture("texture2.png")

    clock = pygame.time.Clock()
    speed = 0.2

    # Режимы: 0 — тетраэдр, 1 — куб с 1 текстурой, 2 — куб с 2 текстурами
    mode = 0
    color_influence = 0.3
    blend_factor = 0.0
    rotation = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                mode = (mode + 1) % 3  # 0 → 1 → 2 → 0

        keys = pygame.key.get_pressed()

        # Движение камеры
        if keys[K_a] or keys[K_LEFT]:   glTranslatef(-speed,0,0)
        if keys[K_d] or keys[K_RIGHT]:  glTranslatef(speed,0,0)
        if keys[K_w] or keys[K_UP]:     glTranslatef(0,speed,0)
        if keys[K_s] or keys[K_DOWN]:   glTranslatef(0,-speed,0)
        if keys[K_PAGEUP]:              glTranslatef(0,0,speed)
        if keys[K_PAGEDOWN]:            glTranslatef(0,0,-speed)

        # Управление параметрами в зависимости от режима
        if mode == 1:  # Куб с одной текстурой
            if keys[K_MINUS]:           color_influence = max(0.0, color_influence - 0.02)
            if keys[K_EQUALS] or keys[K_PLUS]: color_influence = min(1.0, color_influence + 0.02)
        elif mode == 2:  # Куб с двумя текстурами
            if keys[K_LEFTBRACKET]:    blend_factor = max(0.0, blend_factor - 0.02)
            if keys[K_RIGHTBRACKET]:   blend_factor = min(1.0, blend_factor + 0.02)
            # Также можно регулировать цвет и тут
            if keys[K_MINUS]:           color_influence = max(0.0, color_influence - 0.02)
            if keys[K_EQUALS] or keys[K_PLUS]: color_influence = min(1.0, color_influence + 0.02)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glRotatef(rotation, 1, 1, 1)
        rotation += 0.5

        if mode == 0:
            draw_tetrahedron()
        elif mode == 1:
            draw_cube_single_texture(tex1, color_influence)
        elif mode == 2:
            draw_cube_two_textures(tex1, tex2, blend_factor, color_influence)

        glPopMatrix()

        # Заголовок с подсказками
        titles = [
            "1. Тетраэдр",
            f"2. Куб — одна текстура | Цвет: {int(color_influence*100)}% | -/= — цвет",
            f"3. Куб — две текстуры | Цвет: {int(color_influence*100)}% | Текстура2: {int(blend_factor*100)}% | [/] — смешивание | -/= — цвет"
        ]
        pygame.display.set_caption(
            f"{titles[mode]} | SPACE — следующий режим | WASD/стрелки — движение"
        )

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
