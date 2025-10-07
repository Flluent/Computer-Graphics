import numpy as np
import matplotlib.pyplot as plt

class PolygonTransformer:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float)
        self.homogeneous_vertices = self._to_homogeneous(self.vertices)
    
    def _to_homogeneous(self, vertices):
        return np.column_stack([vertices, np.ones(len(vertices))])
    
    def _from_homogeneous(self, homogeneous_vertices):
        return homogeneous_vertices[:, :2]
    
    def get_center(self):
        return np.mean(self.vertices, axis=0)
    
    def translate(self, dx, dy):
        translation_matrix = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])
        
        self.homogeneous_vertices = self.homogeneous_vertices @ translation_matrix.T
        self.vertices = self._from_homogeneous(self.homogeneous_vertices)
    
    def rotate_around_point(self, angle_degrees, point):
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        self.translate(-point[0], -point[1])
        self.homogeneous_vertices = self.homogeneous_vertices @ rotation_matrix.T
        self.translate(point[0], point[1])
    
    def rotate_around_center(self, angle_degrees):
        center = self.get_center()
        self.rotate_around_point(angle_degrees, center)
    
    def scale_around_point(self, scale_x, scale_y, point):
        scale_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])
        
        self.translate(-point[0], -point[1])
        self.homogeneous_vertices = self.homogeneous_vertices @ scale_matrix.T
        self.translate(point[0], point[1])
    
    def scale_around_center(self, scale_x, scale_y):
        center = self.get_center()
        self.scale_around_point(scale_x, scale_y, center)
    
    def get_vertices(self):
        return self.vertices.copy()
    
    def plot(self, ax, color='blue', label='', alpha=1.0):
        vertices = self.get_vertices()
        closed_vertices = np.vstack([vertices, vertices[0]])
        ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], 
                color=color, label=label, alpha=alpha, marker='o')
        ax.fill(closed_vertices[:, 0], closed_vertices[:, 1], 
                color=color, alpha=0.3*alpha)

def demonstrate_transformations():
    original_vertices = [(2, 2), (4, 2), (3, 4)]
    polygon = PolygonTransformer(original_vertices)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    polygon.plot(axes[0], color='blue', label='Исходный')
    axes[0].set_title('Исходный полигон')
    axes[0].grid(True)
    axes[0].legend()
    
    polygon_translated = PolygonTransformer(original_vertices)
    polygon_translated.translate(2, 1)
    polygon_translated.plot(axes[1], color='red', label='Смещение (2,1)')
    axes[1].set_title('Смещение')
    axes[1].grid(True)
    axes[1].legend()
    
    polygon_rotated_point = PolygonTransformer(original_vertices)
    polygon_rotated_point.rotate_around_point(45, (3, 3))
    polygon_rotated_point.plot(axes[2], color='green', label='Поворот 45° вокруг (3,3)')
    axes[2].set_title('Поворот вокруг точки')
    axes[2].grid(True)
    axes[2].legend()
    
    polygon_rotated_center = PolygonTransformer(original_vertices)
    polygon_rotated_center.rotate_around_center(30)
    polygon_rotated_center.plot(axes[3], color='purple', label='Поворот 30° вокруг центра')
    axes[3].set_title('Поворот вокруг центра')
    axes[3].grid(True)
    axes[3].legend()
    
    polygon_scaled_point = PolygonTransformer(original_vertices)
    polygon_scaled_point.scale_around_point(1.5, 0.8, (3, 3))
    polygon_scaled_point.plot(axes[4], color='orange', label='Масштаб (1.5, 0.8) отн. (3,3)')
    axes[4].set_title('Масштабирование относительно точки')
    axes[4].grid(True)
    axes[4].legend()
    
    polygon_scaled_center = PolygonTransformer(original_vertices)
    polygon_scaled_center.scale_around_center(0.7, 1.3)
    polygon_scaled_center.plot(axes[5], color='brown', label='Масштаб (0.7, 1.3) отн. центра')
    axes[5].set_title('Масштабирование относительно центра')
    axes[5].grid(True)
    axes[5].legend()
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_transformations()
    
    square_vertices = [(1, 1), (3, 1), (3, 3), (1, 3)]
    square = PolygonTransformer(square_vertices)
    
    print(f"Исходные вершины:\n{square.get_vertices()}")
    
    square.translate(1, 0.5)
    print(f"После смещения на (1, 0.5):\n{square.get_vertices()}")
    
    square.rotate_around_center(45)
    print(f"После поворота на 45° вокруг центра:\n{square.get_vertices()}")
    
    square.scale_around_point(2, 1.5, (2, 2))
    print(f"После масштабирования (2, 1.5) относительно точки (2,2):\n{square.get_vertices()}")