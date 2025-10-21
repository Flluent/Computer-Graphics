import cv2
import numpy as np


class BezierCurve:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.control_points = []
        self.selected_point = None
        self.dragging = False
        self.point_radius = 8

        # Создаем окно
        self.window_name = "Bezier Curve"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def bezier_point(self, t, p0, p1, p2, p3):
        """Вычисляет точку на кубической кривой Безье для параметра t"""
        u = 1 - t
        u2 = u * u
        u3 = u2 * u
        t2 = t * t
        t3 = t2 * t

        point = (u3 * p0 +
                 3 * u2 * t * p1 +
                 3 * u * t2 * p2 +
                 t3 * p3)
        return point

    def draw_bezier_curve(self, img, p0, p1, p2, p3, color=(0, 255, 0), thickness=2):
        """Рисует сегмент кубической кривой Безье"""
        points = []
        for t in np.linspace(0, 1, 100):
            point = self.bezier_point(t, p0, p1, p2, p3)
            points.append(point.astype(int))

        # Рисуем кривую как ломаную линию
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color, thickness)

    def draw_control_polygon(self, img, color=(255, 0, 0), thickness=1):
        """Рисует контрольный полигон"""
        if len(self.control_points) < 2:
            return

        for i in range(len(self.control_points) - 1):
            cv2.line(img,
                     tuple(self.control_points[i].astype(int)),
                     tuple(self.control_points[i + 1].astype(int)),
                     color, thickness)

    def draw_control_points(self, img):
        """Рисует контрольные точки"""
        for i, point in enumerate(self.control_points):
            color = (0, 0, 255) if i == self.selected_point else (255, 0, 0)
            cv2.circle(img, tuple(point.astype(int)), self.point_radius, color, -1)
            cv2.putText(img, str(i),
                        tuple((point + np.array([10, -10])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def is_point_clicked(self, point, mouse_pos):
        """Проверяет, была ли нажата контрольная точка"""
        return np.linalg.norm(point - mouse_pos) <= self.point_radius

    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        mouse_pos = np.array([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверяем, была ли нажата существующая точка
            point_found = False
            for i, point in enumerate(self.control_points):
                if self.is_point_clicked(point, mouse_pos):
                    self.selected_point = i
                    self.dragging = True
                    point_found = True
                    break

            # Если не нажата существующая точка - добавляем новую
            if not point_found:
                self.control_points.append(mouse_pos.astype(float))
                self.selected_point = len(self.control_points) - 1
                self.dragging = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Удаляем точку при правом клике
            for i, point in enumerate(self.control_points):
                if self.is_point_clicked(point, mouse_pos):
                    self.control_points.pop(i)
                    self.selected_point = None
                    break

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Перемещаем выбранную точку
            if self.selected_point is not None:
                self.control_points[self.selected_point] = mouse_pos.astype(float)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def draw_composite_bezier(self, img):
        """Рисует составную кубическую кривую Безье"""
        if len(self.control_points) < 4:
            return

        # Рисуем сегменты кривой
        for i in range(0, len(self.control_points) - 3, 3):
            p0 = self.control_points[i]
            p1 = self.control_points[i + 1]
            p2 = self.control_points[i + 2]
            p3 = self.control_points[i + 3]

            self.draw_bezier_curve(img, p0, p1, p2, p3)

    def draw(self):
        """Основная функция отрисовки"""
        while True:
            # Создаем чистый холст
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Рисуем контрольный полигон
            self.draw_control_polygon(img)

            # Рисуем составную кривую Безье
            self.draw_composite_bezier(img)

            # Рисуем контрольные точки
            self.draw_control_points(img)

            # Отображаем инструкции
            instructions = [
                "Left Click: Add/Move Point",
                "Right Click: Delete Point",
                "ESC: Exit"
            ]

            for i, text in enumerate(instructions):
                cv2.putText(img, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Показываем количество точек
            cv2.putText(img, f"Points: {len(self.control_points)}",
                        (10, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Отображаем изображение
            cv2.imshow(self.window_name, img)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):  # Очистить все точки
                self.control_points = []
                self.selected_point = None

        cv2.destroyAllWindows()


# Запуск программы
if __name__ == "__main__":
    bezier = BezierCurve()
    bezier.draw()