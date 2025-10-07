import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_grayscale_method1(image):
    return np.dot(image[...,:3], [0.2126, 0.7152, 0.0722])

def rgb_to_grayscale_method2(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

image = cv2.imread("bliss.jpg")

if image is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_image1 = rgb_to_grayscale_method1(image).astype(np.uint8)
gray_image2 = rgb_to_grayscale_method2(image).astype(np.uint8)

difference_image = cv2.absdiff(gray_image1, gray_image2)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Сравнение методов преобразования в оттенки серого', fontsize=16)

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Исходное изображение')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray_image1, cmap='gray')
axes[0, 1].set_title('Метод 1: Среднее арифметическое')
axes[0, 1].axis('off')

axes[0, 2].hist(gray_image1.ravel(), 256, [0, 256], color='gray', alpha=0.7)
axes[0, 2].set_title('Гистограмма метода 1')
axes[0, 2].set_xlabel("Интенсивность")
axes[0, 2].set_ylabel("Количество пикселей")

# Пустое место для выравнивания
axes[1, 0].axis('off')

axes[1, 1].imshow(gray_image2, cmap='gray')
axes[1, 1].set_title('Метод 2: Взвешенное среднее')
axes[1, 1].axis('off')

axes[1, 2].hist(gray_image2.ravel(), 256, [0, 256], color='blue', alpha=0.7)
axes[1, 2].set_title('Гистограмма метода 2')
axes[1, 2].set_xlabel("Интенсивность")
axes[1, 2].set_ylabel("Количество пикселей")

axes[2, 0].axis('off')

axes[2, 1].imshow(difference_image, cmap="gray")
axes[2, 1].set_title('Разница между методами')
axes[2, 1].axis('off')

axes[2, 2].hist(difference_image.ravel(), 256, [0, 256], color='red', alpha=0.7)
axes[2, 2].set_title('Гистограмма разницы')
axes[2, 2].set_xlabel("Интенсивность")
axes[2, 2].set_ylabel("Количество пикселей")


plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()


cv2.destroyAllWindows()
