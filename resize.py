import os
import cv2

path = '/media/mnt/dataset/DeepFashion_image_black_512padding'
lines = os.listdir(path)
print(lines)

for i, imagee in enumerate(lines):
    if len(lines) > 0:
        print(i)
        img = cv2.imread(imagee, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        path = '/media/mnt/dataset/DeepFashion_image_resize' + str(i).zfill(5) + '.png'
        cv2.imwrite(path, img)
