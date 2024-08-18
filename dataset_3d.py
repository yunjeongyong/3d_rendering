import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from option.config import Config
from torchvision import transforms
from models.unet_model import UNet

class THREED_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, seg_path, train_mode, transforms=None, results=None):
        self.path = path
        self.seg_path = seg_path
        self.train_mode = train_mode
        self.transforms = transforms
        self.obj_data_dir = os.path.join(self.path, 'obj_file_deepfashion')

        pred_img_path_list = results['imgname']
        pred_betas = torch.from_numpy(results['shape']).float()
        pred_rotmat = torch.from_numpy(results['pose']).float()
        pred_cam_full = torch.from_numpy(results['global_t']).float()
        pred_joints = torch.from_numpy(results['pred_joints']).float()
        pred_focal_l = torch.from_numpy(results['focal_l']).float()
        pred_detection_all = torch.from_numpy(results['detection_all']).float()
        # print('pred_img_path_list', pred_img_path_list)
        # print('pred_betas', pred_betas)
        # print('pred_rotmat', pred_rotmat)
        # print('pred_cam_full', pred_cam_full)
        # print('pred_joints',pred_joints)
        # print('pred_focal_l',pred_focal_l)
        # print('pred_detection_all',pred_detection_all)



        # obj_name = os.listdir(obj_data_dir)
        # obj_file_full_name = [os.path.join(obj_data_dir, obj) for obj in obj_name]



        object_data = []
        for i in range(len(pred_betas)):
            object_data.append({
                'imgname':pred_img_path_list[i],
                'betas': pred_betas[i, :],
                'rotmat': pred_rotmat[i, :],
                'cam_full': pred_cam_full[i, :],
                'joints': pred_joints[i, :, :],
                'focal_l': pred_focal_l[i],
                'detection_all': pred_detection_all[i, :]
            })
        # print('object_data',object_data)
        l = len(object_data)
        self.train_name = object_data[:int(l * 0.8)]
        self.val_name = object_data[int(l * 0.8):]

    def get_coord(self, shape):
        y = np.linspace(0.0, 1.0, num=shape[0])
        x = np.linspace(0.0, 1.0, num=shape[1])
        coord_y, coord_x = np.meshgrid(y, x, indexing='ij')
        coord = np.concatenate((coord_y[None], coord_x[None]), axis=0)
        return torch.from_numpy(coord).float()


    def __getitem__(self, idx):
        if self.train_mode == True:
            img_name = self.train_name[idx]['imgname']
            # print('img_name',img_name)
            # print('img_name',img_name)
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256))
            # print('img.shape',img.shape)
            # print('img.shape[-2:]',img.shape[-2:])
            # print('img.shape[0,:]', img.shape[0:-1])
            coord = self.get_coord(img.shape[0:-1])
            img_names = '.'.join(img_name.strip().split('.')[:-1])
            n = img_names.split('/')[-1]
            # print('n', n)
            seg_name = self.seg_path + n + '_segm.png'
            # seg_name_ = n + '_segm.png'
            # print('seg_name',seg_name)
            seg = cv2.imread(seg_name, cv2.IMREAD_COLOR)
            seg = cv2.resize(seg, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            # print('img_name',img_name)
            img_name_ = img_name.split('/')[-1]
            img_name_ = os.path.splitext(img_name_)[0]
            obj_name_ = img_name_+'.obj'
            obj_path = os.path.join(self.obj_data_dir, obj_name_)
            # print('obj_path',obj_path)
            # print('img_name_', img_name_)
            # img = ToTensor(img)
            if self.transforms:
                img = self.transforms(img)
                seg = self.transforms(seg)
            self.train_name[idx]['img'] = img
            self.train_name[idx]['coord'] = coord
            # self.train_name[idx]['imgname'] = n
            # self.train_name[idx]['segname'] = seg_name_
            self.train_name[idx]['seg'] = seg
            self.train_name[idx]['obj_path'] = obj_path
            return self.train_name[idx]
        else:
            img_name = self.val_name[idx]['imgname']
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256))
            print('img.shape[-2:]', img.shape[-2:])
            coord = self.get_coord(img.shape[0:-1])
            img_names = '.'.join(img_name.strip().split('.')[:-1])
            # n = img_names.split('/')[-1]
            # print('img_names',img_names)
            seg_name = self.seg_path + img_names.split('/')[-1] + '_segm.png'
            # print('seg_name', seg_name)

            seg_name_ = img_names.split('/')[-1] + '_segm.png'
            # print('seg_name_',seg_name_)
            seg = cv2.imread(seg_name, cv2.IMREAD_COLOR)
            seg = cv2.resize(seg, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            img_name_ = img_name.split('/')[-1]
            img_name_ = os.path.splitext(img_name_)[0]
            obj_name_ = img_name_+'.obj'
            obj_path = os.path.join(self.obj_data_dir, obj_name_)
            # print('obj_path', obj_path)
            if self.transforms:
                img = self.transforms(img)
                seg = self.transforms(seg)
            self.val_name[idx]['img'] = img
            self.val_name[idx]['coord'] = coord
            # self.val_name[idx]['imgname'] = n
            # self.val_name[idx]['segname'] = seg_name_
            self.val_name[idx]['seg'] = seg
            self.val_name[idx]['obj_path'] = obj_path
            return self.val_name[idx]

    def __len__(self):
        if self.train_mode == True:
            return len(self.train_name)
        else:
            return len(self.val_name)


if __name__ == "__main__":
    # config file
    config = Config({
        # device
        "GPU_ID": "2",
        "num_workers": 0,

        # data
        "db_path": "/media/mnt/dataset",
        "seg_path": "/media/mnt/dataset/DeepFashion_segm_black_512padding/",
        # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
        "SMPL_path": "/media/mnt/dataset/DeepFashion_image_black_512pad_cliff_deepfashion_hr48.npz",
        "snap_path": "/media/mnt/Project/data",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "train_size": 0.8,
        "scenes": "all",

        # ensemble in validation phase
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "save_freq": 1,
        "save_freq_model": 5,
        "checkpoint": None,  # load pretrained weights
        "T_max": 50,  # cosine learning rate period (iteration)
        "eta_min": 0  # mininum learning rate
    })

    config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")

    # d_img = torch.rand(1, 3, 512, 512).to(config.device)
    #
    # model1 = UNet(n_channels=3, n_classes=3, bilinear=False).to(config.device)
    # model2 = UNet(n_channels=4, n_classes=3, bilinear=False).to(config.device)
    #
    # texture_feature = model1(d_img).to(config.device)
    # print('texture_feature',texture_feature.shape)
    # # pred, feat_reff = model(d_img, enc_inputs, dec_inputs)
    # dataset = THREED_Dataset(config.db_path)
    results = np.load(config.SMPL_path)
    train_dataset = THREED_Dataset(
        path=config.db_path,
        seg_path=config.seg_path,
        train_mode=True,
        transforms=transforms.Compose([transforms.ToTensor()]),
        results=results
    )

    test_dataset = THREED_Dataset(
        path=config.db_path,
        seg_path=config.seg_path,
        train_mode=False,
        transforms=transforms.Compose([transforms.ToTensor()]),
        results=results
    )

    for i in range(20):
        data = test_dataset[i]
        img = data['img'].to(config.device)
        imgname = data['imgname']
        # segname = data['seg']
        seg = data['seg'].to(config.device)
        betas = data['betas'].to(config.device)
        rotmat = data['rotmat'].to(config.device)
        cam_full = data['cam_full'].to(config.device)
        joints = data['joints'].to(config.device)
        focal_l = data['focal_l'].to(config.device)
        detection_all = data['detection_all'].to(config.device)
        obj_file = data['obj_path']

        print('img', img)
        # new_dir = '/media/mnt/dataset/newnewnew/'
        print('imgname', imgname)
        img_file_name = imgname+'.png'
        # seg_file_name = new_dir + segname
        print('img_file_name', img_file_name)
        # print('seg_file_name', seg_file_name)
        save_image(img, img_file_name)
        # save_image(seg, seg_file_name)
        img = img.permute(1, 2, 0)
        img = img.cpu().detach().numpy()
        plt.imshow(img)
        plt.show()
        # img_file_name = os.path.join(config.db_path, imgname)
        print('img_file_name', img_file_name)
        # cv2.imwrite(img_file_name, img)

        print('img.shape', img.shape)
        print('seg', seg)
        print('seg.shape',seg.shape)
    print(0)