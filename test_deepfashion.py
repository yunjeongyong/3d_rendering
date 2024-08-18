import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = '/media/mnt/dataset/DeepFashion_image_black_512padding'
orig_path = 'E:\dataset\DeepFashion'
lines = os.listdir(path)
print('lines',lines)
image_names = ['.'.join(l.strip().split('.')[:-1]) for l in lines]
print('image_names',image_names)
image_path = [os.path.join(path, 'images', n + '.png') for n in image_names]
print('image_path', image_path)
seg_path = [os.path.join(orig_path, 'segm',n + '_segm.png') for n in image_names]
print('seg_path', seg_path)
os.chdir("/media/mnt/dataset/DeepFashion/segm")
file_list = os.listdir()

print('image_path', len(image_path))
print('seg_path', len(seg_path))


# print('file_list',file_list)
# file_path = []
# for i in file_list:
#     file_name = '/media/mnt/dataset/DeepFashion/segm/' + i
#     print('file_name',file_name)
#     file_path.append(file_name)
# s = set(file_path)
# # print('file_path',file_path)
# different_image_and_seg = [x for x in seg_path if x not in s]
# print('len', len(different_image_and_seg))
# print('different_image_and_seg',different_image_and_seg)
# # image_path_split = image_path[0].split('/')[-1]
# # print('image_path_split',image_path_split)
#


# for i, imagee in enumerate(image_path):
#     if len(image_path) > 0:
#         print(i)
#         print('imagee', imagee)
#         segmm = seg_path[i]
#         print('segmm', segmm)
#         seg = cv2.imread(segmm, cv2.IMREAD_COLOR)
#         img = cv2.imread(imagee, cv2.IMREAD_COLOR)
#         seg = cv2.resize(seg, (256, 512))
#         img = cv2.resize(img, (256, 512))
#         # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # segRGB = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
#
#         mask = seg.sum(-1) == 0
#         img[mask,:] = 0
#
#         save_file = '/media/mnt/dataset/DeepFashion_segm_black_512padding/'
#         h, w, _ = seg.shape
#         best_image = 512
#         new_h = best_image - h
#         new_w = best_image - w
#         top, bottom = new_h // 2, new_h - (new_h // 2)
#         left, right = new_w // 2, new_w - (new_w // 2)
#         new_img = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         save_file_name = save_file + imagee.split('/')[-1]
#         # black_path = '/media/mnt/dsave_file_nameataset/DeepFashion_image_black/' + imagee.split('/')[-1]
#         print('imagee', imagee)
#         # print('black_path', black_path)
#         print('save_file_name',save_file_name)
#         plt.imshow(new_img)
#         plt.show()
#         cv2.imwrite(save_file_name, new_img)

for i, segg in enumerate(seg_path):
    if len(seg_path) > 0:
        print(i)
        print('segg', segg)
        # segmm = seg_path[i]
        # print('segmm', segmm)
        _seg = cv2.imread(segg, cv2.IMREAD_COLOR)
        # img = cv2.imread(imagee, cv2.IMREAD_COLOR)
        _seg = cv2.resize(_seg, (256, 512))
        # img = cv2.resize(img, (256, 512))
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # segRGB = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

        # mask = seg.sum(-1) == 0
        # img[mask,:] = 0

        save_file = '/media/mnt/dataset/DeepFashion_segm_black_512padding/'
        h, w, _ = _seg.shape
        best_image = 512
        new_h = best_image - h
        new_w = best_image - w
        top, bottom = new_h // 2, new_h - (new_h // 2)
        left, right = new_w // 2, new_w - (new_w // 2)
        new_segg = cv2.copyMakeBorder(_seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        save_file_name = save_file + segg.split('/')[-1]
        # black_path = '/media/mnt/dsave_file_nameataset/DeepFashion_image_black/' + imagee.split('/')[-1]
        print('segg', segg)
        # print('black_path', black_path)
        print('save_file_name',save_file_name)
        plt.imshow(new_segg)
        plt.show()
        cv2.imwrite(save_file_name, new_segg)


# for i, image in enumerate(file_list):
#     img = cv2.imread(image, cv2.IMREAD_COLOR)
#     h, w, _ = img.shape
#     best_image = 512
#     new_h = best_image - h
#     new_w = best_image - w
#     top, bottom = new_h//2, new_h - (new_h // 2)
#     left, right = new_w // 2, new_w - (new_w // 2)
#     new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
#     save_file_name = os.path.join(save_file, image)
#     cv2.imwrite(save_file_name, new_img)
