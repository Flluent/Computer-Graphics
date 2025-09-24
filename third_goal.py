import cv2
import numpy as np


def adjust_hsv(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: не удалось загрузить изображение")
        return


    cv2.namedWindow('HSV Adjustment', cv2.WINDOW_NORMAL)


    cv2.createTrackbar('Hue', 'HSV Adjustment', 0, 360, lambda x: None)
    cv2.createTrackbar('Saturation', 'HSV Adjustment', 100, 200, lambda x: None)
    cv2.createTrackbar('Value', 'HSV Adjustment', 100, 200, lambda x: None)

    while True:

        hue_shift = cv2.getTrackbarPos('Hue', 'HSV Adjustment') - 180
        sat_scale = cv2.getTrackbarPos('Saturation', 'HSV Adjustment') / 100.0
        val_scale = cv2.getTrackbarPos('Value', 'HSV Adjustment') / 100.0


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)


        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)


        hsv = hsv.astype(np.uint8)
        adjusted_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


        cv2.imshow('HSV Adjustment', adjusted_img)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = f'adjusted_h{hue_shift}_s{sat_scale}_v{val_scale}.jpg'
            cv2.imwrite(filename, adjusted_img)
            print(f"Сохранено: {filename}")
        elif key == ord('r'):
            cv2.setTrackbarPos('Hue', 'HSV Adjustment', 180)
            cv2.setTrackbarPos('Saturation', 'HSV Adjustment', 100)
            cv2.setTrackbarPos('Value', 'HSV Adjustment', 100)
        elif key == 27:
            break

    cv2.destroyAllWindows()


adjust_hsv('bliss.jpg')