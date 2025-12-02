import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
import math

def check_shader_compile(shader):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if result != GL_TRUE:
        info_log = glGetShaderInfoLog(shader)
        raise Exception(f"Ошибка компиляции шейдера: {info_log}")

def check_program_link(program):
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if result != GL_TRUE:
        info_log = glGetProgramInfoLog(program)
        raise Exception(f"Ошибка линковки программы: {info_log}")

def create_quad():
    vertices = np.array([
        -0.4, -0.4, 0.0,
         0.4, -0.4, 0.0,
         0.4,  0.4, 0.0,
        -0.4, -0.4, 0.0,
         0.4,  0.4, 0.0,
        -0.4,  0.4, 0.0,
    ], dtype=np.float32)
    
    return vertices, 6

def create_fan():
    radius = 0.4
    vertices = []
    
    angles = [0, 120, 240]
    
    for i in range(3):
        angle1 = math.radians(angles[i])
        angle2 = math.radians(angles[i] + 120)
        
        vertices.extend([0.0, 0.0, 0.0])
        
        x1 = radius * math.cos(angle1)
        y1 = radius * math.sin(angle1)
        vertices.extend([x1, y1, 0.0])
        
        x2 = radius * math.cos(angle2)
        y2 = radius * math.sin(angle2)
        vertices.extend([x2, y2, 0.0])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    return vertices, 9

def create_pentagon():
    center = [0.0, 0.0, 0.0]
    radius = 0.4
    num_sides = 5
    
    vertices = []
    
    for i in range(num_sides):
        angle1 = 2 * math.pi * i / num_sides - math.pi/2
        angle2 = 2 * math.pi * (i + 1) / num_sides - math.pi/2
        
        vertices.extend([center[0], center[1], center[2]])
        
        x1 = center[0] + radius * math.cos(angle1)
        y1 = center[1] + radius * math.sin(angle1)
        vertices.extend([x1, y1, 0.0])
        
        x2 = center[0] + radius * math.cos(angle2)
        y2 = center[1] + radius * math.sin(angle2)
        vertices.extend([x2, y2, 0.0])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    return vertices, 15

def main():
    if not glfw.init():
        print("Ошибка: Не удалось инициализировать GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Лаб 1: Фигуры и плоское закрашивание (шейдер)", None, None)
    if not window:
        print("Ошибка: Не удалось создать окно GLFW")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    def framebuffer_size_callback(window, width, height):
        glViewport(0, 0, width, height)
    
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glViewport(0, 0, 800, 600)
    
    current_figure = 1
    
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
        FragColor = vec4(0.8, 0.3, 0.2, 1.0);
    }
    """
    
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    check_shader_compile(vertex_shader)
    
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    check_shader_compile(fragment_shader)
    
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    check_program_link(shader_program)
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    quad_vertices, quad_vertex_count = create_quad()
    fan_vertices, fan_vertex_count = create_fan()
    pentagon_vertices, pentagon_vertex_count = create_pentagon()
    
    VAOs = []
    VBOs = []
    
    all_vertices = [quad_vertices, fan_vertices, pentagon_vertices]
    
    for vertices_data in all_vertices:
        VAO = glGenVertexArrays(1)
        VAOs.append(VAO)
        
        glBindVertexArray(VAO)
        
        VBO = glGenBuffers(1)
        VBOs.append(VBO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    vertex_counts = [quad_vertex_count, fan_vertex_count, pentagon_vertex_count]
    
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    
    print("Задания: 1. Рисование фигур, 2. Плоское закрашивание (цвет в шейдере)")
    print("\nУправление:")
    print("1 - четырехугольник")
    print("2 - веер (3 треугольника)")
    print("3 - пятиугольник")
    print("ESC - выход")
    
    figure_names = ["четырехугольник", "веер", "пятиугольник"]
    
    while not glfw.window_should_close(window):
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
            current_figure = 1
        elif glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
            current_figure = 2
        elif glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
            current_figure = 3
        
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(shader_program)
        glBindVertexArray(VAOs[current_figure - 1])
        glDrawArrays(GL_TRIANGLES, 0, vertex_counts[current_figure - 1])
        glBindVertexArray(0)
        glUseProgram(0)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    for VAO in VAOs:
        glDeleteVertexArrays(1, [VAO])
    
    for VBO in VBOs:
        glDeleteBuffers(1, [VBO])
    
    glDeleteProgram(shader_program)
    glfw.terminate()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        glfw.terminate()