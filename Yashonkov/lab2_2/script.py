import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('bliss.JPG')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r_channel = image_rgb[:, :, 0]  # Красный канал
g_channel = image_rgb[:, :, 1]  # Зеленый канал
b_channel = image_rgb[:, :, 2]  # Синий канал

r_image = np.zeros_like(image_rgb)
r_image[:, :, 0] = r_channel  # Красный канал

g_image = np.zeros_like(image_rgb)
g_image[:, :, 1] = g_channel  # Зеленый канал

b_image = np.zeros_like(image_rgb)
b_image[:, :, 2] = b_channel  # Синий канал

# Построение гистограмм
plt.figure(figsize=(15, 10))

# Исходное изображение
plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

# Красный канал
plt.subplot(2, 4, 2)
plt.imshow(r_image)
plt.title('Красный канал (R)')
plt.axis('off')

# Зеленый канал
plt.subplot(2, 4, 3)
plt.imshow(g_image)
plt.title('Зеленый канал (G)')
plt.axis('off')

# Синий канал
plt.subplot(2, 4, 4)
plt.imshow(b_image)
plt.title('Синий канал (B)')
plt.axis('off')

# Гистограмма красного канала
plt.subplot(2, 4, 5)
plt.hist(r_channel.ravel(), bins=256, color='red', alpha=0.7)
plt.title('Гистограмма красного канала')
plt.xlabel('Значение пикселя')
plt.ylabel('Частота')
plt.xlim(0, 255)

# Зеленого канала
plt.subplot(2, 4, 6)
plt.hist(g_channel.ravel(), bins=256, color='green', alpha=0.7)
plt.title('Гистограмма зеленого канала')
plt.xlabel('Значение пикселя')
plt.ylabel('Частота')
plt.xlim(0, 255)

# Синего канала
plt.subplot(2, 4, 7)
plt.hist(b_channel.ravel(), bins=256, color='blue', alpha=0.7)
plt.title('Гистограмма синего канала')
plt.xlabel('Значение пикселя')
plt.ylabel('Частота')
plt.xlim(0, 255)

# Совмещенная
plt.subplot(2, 4, 8)
plt.hist(r_channel.ravel(), bins=256, color='red', alpha=0.5, label='Red')
plt.hist(g_channel.ravel(), bins=256, color='green', alpha=0.5, label='Green')
plt.hist(b_channel.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
plt.title('Совмещенная гистограмма')
plt.xlabel('Значение пикселя')
plt.ylabel('Частота')
plt.xlim(0, 255)
plt.legend()

plt.tight_layout()
plt.show()

# Вывод
print("Статистика по каналам:")
print(f"Красный канал: min={r_channel.min()}, max={r_channel.max()}, mean={r_channel.mean():.2f}")
print(f"Зеленый канал: min={g_channel.min()}, max={g_channel.max()}, mean={g_channel.mean():.2f}")
print(f"Синий канал: min={b_channel.min()}, max={b_channel.max()}, mean={b_channel.mean():.2f}")