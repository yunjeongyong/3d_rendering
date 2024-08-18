import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# path = '/media/mnt/dataset/DeepFashion_image_black_512padding'
# orig_path = '/media/mnt/dataset'
# lines = os.listdir(orig_path)
path = '/media/mnt/dataset/AHP/train/ImageSets/train.txt'
img_path = '/media/mnt/dataset/ahp_padding_img'
seg_path = '/media/mnt/dataset/ahp_padding_seg'
# orig_path = 'E:\dataset\DeepFashion'
# lines = os.listdir(path)
# print('lines',lines)

txt_name = []
f = open(path, 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    txt_name.append(line)
f.close()
print('txt_name',txt_name)

image_path = [os.path.join(img_path, n + '.jpg')for n in txt_name]
seg_path = [os.path.join(seg_path, n + '.png')for n in txt_name]
print('image_path', image_path)
print('seg_path', seg_path)
print('image_path', len(image_path))
print('seg_path', len(seg_path))
# print('lines',lines)
# _lines = [s for s in lines if "WOMEN" in s]
# _lines = [s for s in lines if "MEN" in s]
# print('_lines',_lines)
# image_names = ['.'.join(l.strip().split('.')[:-1]) for l in _lines]
# print('image_names',image_names)
# image_path = [os.path.join(orig_path, n + '.jpg') for n in image_names]
# print('image_path', image_path)
# image_names = ['.'.join(l.strip().split('.')[:-1]) for l in lines]
# print('image_names',image_names)
# image_path = [os.path.join(orig_path,'images', n + '.jpg') for n in image_names]
# print('image_path', image_path)
# seg_path = [os.path.join(orig_path, 'segm',n + '_segm.png') for n in image_names]
# print('seg_path', seg_path)
#
# print('image_names', len(image_names))
# print('image_path', len(image_path))
# print('seg_path', len(seg_path))


# os.chdir("/media/mnt/dataset/DeepFashion/segm")
# file_list = os.listdir()
#
# print('image_path', len(image_path))
# print('seg_path', len(seg_path))
#
# for i in image_path:
#     i = i.split('/')[-1]
#     print('i',i)
#     subprocess.call(["mv /media/mnt/dataset/"+i+ " /media/mnt/dataset/DeepFashion_cliff_result/"], shell=True)
#     #shell=True하니까 돌아가네..? 뭐냐..

# # print('file_list',file_list)
# # file_path = []
# # for i in file_list:
# #     file_name = '/media/mnt/dataset/DeepFashion/segm/' + i
# #     print('file_name',file_name)
# #     file_path.append(file_name)
# # s = set(file_path)
# # # print('file_path',file_path)
# # different_image_and_seg = [x for x in seg_path if x not in s]
# # print('len', len(different_image_and_seg))
# # print('different_image_and_seg',different_image_and_seg)
# # # image_path_split = image_path[0].split('/')[-1]
# # # print('image_path_split',image_path_split)
# #
#
#
for i, imagee in enumerate(image_path):
    if len(image_path) > 0:
        print(i)
        print('imagee', imagee)
        segmm = seg_path[i]
        print('segmm', segmm)
        seg = cv2.imread(segmm, cv2.IMREAD_COLOR)
        img = cv2.imread(imagee, cv2.IMREAD_COLOR)
        # seg = cv2.resize(seg, (256, 512))
        # img = cv2.resize(img, (256, 512))
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # segRGB = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

        mask = seg.sum(-1) == 0
        img[mask,:] = 0

        save_file = '/media/mnt/dataset/ahp_segm_black_512padding/'
        # h, w, _ = seg.shape
        # best_image = 512
        # new_h = best_image - h
        # new_w = best_image - w
        # top, bottom = new_h // 2, new_h - (new_h // 2)
        # left, right = new_w // 2, new_w - (new_w // 2)
        # new_img = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        save_file_name = save_file + imagee.split('/')[-1]
        # black_path = '/media/mnt/dsave_file_nameataset/DeepFashion_image_black/' + imagee.split('/')[-1]
        print('imagee', imagee)
        # print('black_path', black_path)
        print('save_file_name',save_file_name)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(save_file_name, img)
#
# for i, segg in enumerate(seg_path):
#     if len(seg_path) > 0:
#         print(i)
#         print('segg', segg)
#         # segmm = seg_path[i]
#         # print('segmm', segmm)
#         _seg = cv2.imread(segg, cv2.IMREAD_COLOR)
#         # img = cv2.imread(imagee, cv2.IMREAD_COLOR)
#         # _seg = cv2.resize(_seg, 512)
#         # img = cv2.resize(img, (256, 512))
#         # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # segRGB = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
#
#         # mask = seg.sum(-1) == 0
#         # img[mask,:] = 0
#
#         save_file = '/media/mnt/dataset/ahp_padding_seg/'
#         h, w, _ = _seg.shape
#         print('h', h)
#         print('w', w)
#         if h > w:
#             best_image = h
#         elif h == w:
#             best_image = h
#         else:
#             best_image = w
#
#         new_h = best_image - h
#         new_w = best_image - w
#         top, bottom = new_h // 2, new_h - (new_h // 2)
#         left, right = new_w // 2, new_w - (new_w // 2)
#         new_segg = cv2.copyMakeBorder(_seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         new_seggg = cv2.resize(new_segg, (512, 512))
#         save_file_name = save_file + segg.split('/')[-1]
#         # black_path = '/media/mnt/dsave_file_nameataset/DeepFashion_image_black/' + imagee.split('/')[-1]
#         print('segg', segg)
#         # print('black_path', black_path)
#         print('save_file_name',save_file_name)
#         plt.imshow(new_segg)
#         plt.show()
#         cv2.imwrite(save_file_name, new_seggg)
#
#
# # for i, image in enumerate(file_list):
# #     img = cv2.imread(image, cv2.IMREAD_COLOR)
# #     h, w, _ = img.shape
# #     best_image = 512
# #     new_h = best_image - h
# #     new_w = best_image - w
# #     top, bottom = new_h//2, new_h - (new_h // 2)
# #     left, right = new_w // 2, new_w - (new_w // 2)
# #     new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
# #     save_file_name = os.path.join(save_file, image)
# #     cv2.imwrite(save_file_name, new_img)
