import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import os

path = '/media/mnt/dataset/ahp_testtt'
result_path = '/media/mnt/dataset/ahp_crop_512'

os.chdir(path)
files = os.listdir(path)
print(files)

# png_img = []
# png_img_gray = []
# jpg_img = []
# jpg_img_gray = []
for file in files:
    if '.png' in file:
        f = cv2.imread(file, cv2.IMREAD_COLOR)
        d = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img_names = '.'.join(file.strip().split('.')[:-1])
        # n = img_names.split('/')[-1]
        blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
        ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

        # cv2.imwrite('image.png', image)
        # cv2.imwrite('image_gray.png', image_gray)

        edged = cv2.Canny(blur, 10, 250)
        # cv2.imwrite('edged.png', edged)
        # cv2.imshow('Edged', edged)
        # cv2.waitKey(0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite('closed.png', closed)
        # cv2.imshow('closed', closed)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0

        # 초록색 선 생김
        # contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
        # cv2.imwrite('contours_image.png', contours_image)
        # cv2.imshow('contours_image', contours_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours_xy = np.array(contours)
        contours_xy.shape

        x_min, x_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
                x_min = min(value)
                x_max = max(value)
        print(x_min)
        print(x_max)

        # y의 min과 max 찾기
        y_min, y_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
                y_min = min(value)
                y_max = max(value)
        print(y_min)
        print(y_max)

        if x_min == 0 or x_max == 0 or y_min == 0 or y_max == 0:
            pass
        else:
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            img_trim = image[y:y + h, x:x + w]
            # img_trim = cv2.resize(img_trim, (512, 512))
            cv2.imwrite(os.path.join(result_path, '{}.jpg'.format(img_names)), img_trim)
        # png_img.append(f)
        # png_img_gray.append(d)

    if '.jpg' in file:
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img_names = '.'.join(file.strip().split('.')[:-1])
        print("img_name", img_names)
        # n = img_names.split('/')[-1]
        blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
        ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

        # cv2.imwrite('image.png', image)
        # cv2.imwrite('image_gray.png', image_gray)

        edged = cv2.Canny(blur, 10, 250)
        # cv2.imwrite('edged.png', edged)
        # cv2.imshow('Edged', edged)
        # cv2.waitKey(0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite('closed.png', closed)
        # cv2.imshow('closed', closed)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = 0

        # 초록색 선 생김
        # contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
        # cv2.imwrite('contours_image.png', contours_image)
        # cv2.imshow('contours_image', contours_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours_xy = np.array(contours)
        contours_xy.shape

        x_min, x_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
                x_min = min(value)
                x_max = max(value)
        print(x_min)
        print(x_max)

        # y의 min과 max 찾기
        y_min, y_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
                y_min = min(value)
                y_max = max(value)
        print(y_min)
        print(y_max)

        if x_min == 0 or x_max == 0 or y_min == 0 or y_max == 0:
            pass
        else:
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            img_trim = image[y:y + h, x:x + w]
            # img_trim = cv2.resize(img_trim, (512, 512))
            cv2.imwrite(os.path.join(result_path,'{}.jpg'.format(img_names)), img_trim)


# image = cv2.imread(path, cv2.IMREAD_COLOR)
# image_2 = cv2.imread(path, cv2.IMREAD_COLOR)
# image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# # b, g, r = cv2.split(image)
# # image2 = cv2.merge([r, g, b])
#
# # blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
# # ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
#
# blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
# ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
#
# # cv2.imwrite('image.png', image)
# # cv2.imwrite('image_gray.png', image_gray)
#
# edged = cv2.Canny(blur, 10, 250)
# # cv2.imwrite('edged.png', edged)
# # cv2.imshow('Edged', edged)
# # cv2.waitKey(0)
#
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# # cv2.imwrite('closed.png', closed)
# # cv2.imshow('closed', closed)
# # cv2.waitKey(0)
#
# contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# total = 0
#
# # 초록색 선 생김
# # contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
# # cv2.imwrite('contours_image.png', contours_image)
# # cv2.imshow('contours_image', contours_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# contours_xy = np.array(contours)
# contours_xy.shape
#
# x_min, x_max = 0, 0
# value = list()
# for i in range(len(contours_xy)):
#     for j in range(len(contours_xy[i])):
#         value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
#         x_min = min(value)
#         x_max = max(value)
# print(x_min)
# print(x_max)
#
# # y의 min과 max 찾기
# y_min, y_max = 0, 0
# value = list()
# for i in range(len(contours_xy)):
#     for j in range(len(contours_xy[i])):
#         value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
#         y_min = min(value)
#         y_max = max(value)
# print(y_min)
# print(y_max)
#
# x = x_min
# y = y_min
# w = x_max-x_min
# h = y_max-y_min
# img_trim = image_2[y:y+h, x:x+w]
# img_trim = cv2.resize(img_trim, (512, 512))
# cv2.imwrite('org_trim.jpg', img_trim)
