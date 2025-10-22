import turtle
import random
import math


class LSystem:
    def __init__(self):
        self.screen = turtle.Screen()
        self.screen.setup(1000, 800)
        self.screen.bgcolor("white")
        self.screen.tracer(0, 0)

        self.t = turtle.Turtle()
        self.t.speed(0)
        self.t.hideturtle()

        self.axiom = ""
        self.angle = 0
        self.start_angle = 0
        self.rules = {}
        self.distance = 10
        self.random_factor = 0.0

    def load_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                return False

            first_line = lines[0].split()
            self.axiom = first_line[0]
            self.angle = float(first_line[1])

            if len(first_line) > 2:
                self.start_angle = float(first_line[2])
            else:
                self.start_angle = 90

            self.rules = {}
            for line in lines[1:]:
                if '->' in line:
                    key, value = line.split('->')
                    self.rules[key.strip()] = value.strip()
            return True
        except Exception as e:
            print(f"Ошибка загрузки файла {filename}: {e}")
            return False

    def set_randomness(self, factor):
        self.random_factor = factor

    def generate_string(self, iterations):
        current_string = self.axiom

        for _ in range(iterations):
            new_string = ""
            for char in current_string:
                if char in self.rules:
                    rule = self.rules[char]
                    if '|' in rule and self.random_factor > 0:
                        options = rule.split('|')
                        if random.random() < self.random_factor:
                            new_string += random.choice(options)
                        else:
                            new_string += options[0]
                    else:
                        new_string += rule
                else:
                    new_string += char
            current_string = new_string

        return current_string

    def calculate_bounds(self, commands):
        stack = []
        x, y = 0, 0
        angle = self.start_angle
        min_x = max_x = min_y = max_y = 0

        for cmd in commands:
            if cmd == 'F' or cmd == 'G':
                rad_angle = math.radians(angle)
                x += self.distance * math.cos(rad_angle)
                y += self.distance * math.sin(rad_angle)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            elif cmd == '+':
                angle -= self.angle
            elif cmd == '-':
                angle += self.angle
            elif cmd == '[':
                stack.append((x, y, angle))
            elif cmd == ']':
                if stack:
                    x, y, angle = stack.pop()

        return min_x, max_x, min_y, max_y

    def draw_fractal(self, commands, scale=1.0):
        self.t.clear()
        self.t.penup()
        self.t.goto(0, 0)
        self.t.setheading(self.start_angle)
        self.t.pendown()
        self.t.pencolor("black")

        stack = []
        scaled_distance = self.distance * scale

        for cmd in commands:
            if cmd == 'F' or cmd == 'G':
                actual_distance = scaled_distance
                if self.random_factor > 0:
                    actual_distance *= random.uniform(0.8, 1.2)
                self.t.forward(actual_distance)

            elif cmd == 'f':
                actual_distance = scaled_distance
                if self.random_factor > 0:
                    actual_distance *= random.uniform(0.8, 1.2)
                self.t.penup()
                self.t.forward(actual_distance)
                self.t.pendown()

            elif cmd == '+':
                actual_angle = self.angle
                if self.random_factor > 0:
                    actual_angle += random.uniform(-self.angle * self.random_factor,
                                                   self.angle * self.random_factor)
                self.t.right(actual_angle)

            elif cmd == '-':
                actual_angle = self.angle
                if self.random_factor > 0:
                    actual_angle += random.uniform(-self.angle * self.random_factor,
                                                   self.angle * self.random_factor)
                self.t.left(actual_angle)

            elif cmd == '[':
                stack.append((self.t.xcor(), self.t.ycor(), self.t.heading()))

            elif cmd == ']':
                if stack:
                    x, y, heading = stack.pop()
                    self.t.penup()
                    self.t.goto(x, y)
                    self.t.setheading(heading)
                    self.t.pendown()

        self.screen.update()

    def draw_tree(self, commands, iterations, scale=1.0):
        self.t.clear()
        self.t.penup()
        self.t.goto(0, -300)
        self.t.setheading(90)
        self.t.pendown()

        stack = []
        scaled_distance = self.distance * scale

        base_pensize = 15

        for cmd in commands:
            current_depth = len(stack)
            depth_ratio = current_depth / max(iterations, 1)

            pensize = base_pensize * (1 - depth_ratio * 0.7)
            self.t.pensize(max(pensize, 1))

            if depth_ratio < 0.4:
                r = int(101 - (101 - 60) * depth_ratio / 0.4)
                g = int(67 - (67 - 120) * depth_ratio / 0.4)
                b = int(33 + (80 - 33) * depth_ratio / 0.4)
            else:
                green_ratio = (depth_ratio - 0.4) / 0.6
                r = int(60 - 30 * green_ratio)
                g = int(120 + 50 * green_ratio)
                b = int(80 + 40 * green_ratio)

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            self.t.pencolor(r / 255, g / 255, b / 255)

            if cmd == 'F' or cmd == 'G':
                actual_distance = scaled_distance * (1 - depth_ratio * 0.4)
                if self.random_factor > 0:
                    actual_distance *= random.uniform(0.7, 1.3)
                self.t.forward(actual_distance)

            elif cmd == '+':
                actual_angle = self.angle
                if self.random_factor > 0:
                    actual_angle += random.uniform(-20, 20)
                self.t.right(actual_angle)

            elif cmd == '-':
                actual_angle = self.angle
                if self.random_factor > 0:
                    actual_angle += random.uniform(-20, 20)
                self.t.left(actual_angle)

            elif cmd == '[':
                stack.append((self.t.xcor(), self.t.ycor(), self.t.heading(),
                              self.t.pensize(), scaled_distance))
                scaled_distance *= 0.65

            elif cmd == ']':
                if stack:
                    x, y, heading, pensize, old_distance = stack.pop()
                    self.t.penup()
                    self.t.goto(x, y)
                    self.t.setheading(heading)
                    self.t.pensize(pensize)
                    scaled_distance = old_distance
                    self.t.pendown()

        self.screen.update()


def create_example_files():
    with open('koch.txt', 'w', encoding='utf-8') as f:
        f.write("F 60 0\n")
        f.write("F -> F-F++F-F\n")

    with open('koch_island.txt', 'w', encoding='utf-8') as f:
        f.write("F+F+F+F 90 0\n")
        f.write("F -> F+F-F-FF+F+F-F\n")

    with open('sierpinski.txt', 'w', encoding='utf-8') as f:
        f.write("FXF--FF--FF 60 0\n")
        f.write("F -> FF\n")
        f.write("X -> --FXF++FXF++FXF--\n")

    with open('dragon.txt', 'w', encoding='utf-8') as f:
        f.write("FX 90 0\n")
        f.write("F -> F\n")
        f.write("X -> X+YF+\n")
        f.write("Y -> -FX-Y\n")

    with open('tree_simple.txt', 'w', encoding='utf-8') as f:
        f.write("F 25\n")
        f.write("F -> F[+F]F[-F]F\n")

    with open('tree_complex.txt', 'w', encoding='utf-8') as f:
        f.write("F 22.5\n")
        f.write("F -> FF-[-F+F+F]+[+F-F-F]\n")


def main():
    create_example_files()

    lsys = LSystem()

    while True:
        print("\n" + "=" * 60)
        print("L-СИСТЕМЫ - ФРАКТАЛЬНЫЕ УЗОРЫ И ДЕРЕВЬЯ")
        print("=" * 60)
        print("ФРАКТАЛЬНЫЕ УЗОРЫ (часть 1.а):")
        print("1. Кривая Коха")
        print("2. Квадратный остров Коха")
        print("3. Ковер Серпинского")
        print("4. Кривая дракона")
        print("\nФРАКТАЛЬНЫЕ ДЕРЕВЬЯ (часть 1.б):")
        print("5. Простое дерево")
        print("6. Сложное дерево")
        print("7. Выход")

        choice = input("\nВыберите фрактал (1-7): ").strip()

        if choice == '7':
            break

        if choice in ['1', '2', '3', '4']:
            if choice == '1':
                filename = 'koch.txt'
                iterations = 4
                lsys.distance = 8
                randomness = 0.0
            elif choice == '2':
                filename = 'koch_island.txt'
                iterations = 3
                lsys.distance = 6
                randomness = 0.0
            elif choice == '3':
                filename = 'sierpinski.txt'
                iterations = 5
                lsys.distance = 5
                randomness = 0.0
            else:
                filename = 'dragon.txt'
                iterations = 10
                lsys.distance = 4
                randomness = 0.0

            if not lsys.load_from_file(filename):
                print(f"Ошибка: файл {filename} не найден!")
                continue

            lsys.set_randomness(randomness)

            commands = lsys.generate_string(iterations)
            print(f"Сгенерировано команд: {len(commands)}")

            min_x, max_x, min_y, max_y = lsys.calculate_bounds(commands)
            width = max_x - min_x
            height = max_y - min_y

            if width > 0 and height > 0:
                scale_x = 800 / width
                scale_y = 600 / height
                scale = min(scale_x, scale_y) * 0.8
            else:
                scale = 1.0

            lsys.screen.title(f"L-система: {filename} (итераций: {iterations})")
            lsys.draw_fractal(commands, scale)

        elif choice in ['5', '6']:
            if choice == '5':
                filename = 'tree_simple.txt'
                iterations = 4
                lsys.distance = 20
                randomness = 0.2
            else:
                filename = 'tree_complex.txt'
                iterations = 4
                lsys.distance = 15
                randomness = 0.3

            if not lsys.load_from_file(filename):
                print(f"Ошибка: файл {filename} не найден!")
                continue

            lsys.set_randomness(randomness)

            commands = lsys.generate_string(iterations)
            print(f"Сгенерировано команд для дерева: {len(commands)}")

            lsys.screen.title(f"Фрактальное дерево: {filename}")
            lsys.draw_tree(commands, iterations, scale=1.0)

        else:
            print("Неверный выбор!")
            continue

        print("\nОтрисовка завершена!")
        print("Нажмите в окне turtle для продолжения...")
        lsys.screen.exitonclick()

        lsys.screen = turtle.Screen()
        lsys.screen.setup(1000, 800)
        lsys.screen.bgcolor("white")
        lsys.screen.tracer(0, 0)
        lsys.t = turtle.Turtle()
        lsys.t.speed(0)
        lsys.t.hideturtle()

    turtle.bye()
    print("Программа завершена.")


if __name__ == "__main__":
    main()