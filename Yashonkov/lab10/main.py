import glfw
from OpenGL.GL import *
import numpy as np

# ================= ШЕЙДЕРЫ =================
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(0.0, 1.0, 0.0, 1.0); // зеленый цвет
}
"""

# ================= ФУНКЦИИ =================
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Проверка ошибок
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not(result):
        info_log = glGetShaderInfoLog(shader)
        shader_type_str = "VERTEX" if shader_type==GL_VERTEX_SHADER else "FRAGMENT"
        print(f"ERROR::{shader_type_str}::SHADER_COMPILATION\n{info_log.decode()}")
    return shader

def create_shader_program(vertex_src, fragment_src):
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Проверка линковки
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not(result):
        info_log = glGetProgramInfoLog(program)
        print(f"ERROR::PROGRAM_LINKING\n{info_log.decode()}")

    # Шейдеры больше не нужны после линковки
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

# ================= ИНИЦИАЛИЗАЦИЯ =================
if not glfw.init():
    raise Exception("GLFW не инициализирован")

window = glfw.create_window(800, 600, "Зелёный треугольник", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW окно не создано")

glfw.make_context_current(window)

# Данные треугольника
triangle_vertices = np.array([
    [ 0.0,  0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0]
], dtype=np.float32)

# Создание VBO
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, triangle_vertices.nbytes, triangle_vertices, GL_STATIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, 0)

# Создание шейдерной программы
shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

# Получение атрибута позиции
glUseProgram(shader_program)
position_loc = glGetAttribLocation(shader_program, "aPos")
if position_loc == -1:
    raise Exception("Не удалось получить атрибут aPos")

# ================= ОСНОВНОЙ ЦИКЛ =================
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClearColor(0.0, 0.0, 0.0, 1.0)  # черный фон
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(shader_program)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glEnableVertexAttribArray(position_loc)
    glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    glDrawArrays(GL_TRIANGLES, 0, 3)

    glDisableVertexAttribArray(position_loc)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glfw.swap_buffers(window)

# ================= ОЧИСТКА РЕСУРСОВ =================
glDeleteProgram(shader_program)
glDeleteBuffers(1, [VBO])
glfw.terminate()
