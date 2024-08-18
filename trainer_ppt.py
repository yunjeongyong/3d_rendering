import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# from model.model_2 import Model
from models.unet_model import UNet
from option.config import Config
from dataset import THREED_Dataset
from train_ppt import train_epoch, eval_epoch
from models.smpl import SMPL
from common import constants
import pickle
import numpy as np


# config file
config = Config({
    # device
    "GPU_ID": "2",
    "num_workers": 0,


    # data
    "db_path":"/media/mnt/dataset",
    # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
    "SMPL_path":"/media/mnt/dataset/image_pad_cliff_padding_hr48.npz",
    "snap_path": "/media/mnt/Project/data",  # path for saving weights
    "save_img_path":"/media/mnt/Project/data/rgb_img",
    "ra_body_path":"/media/mnt/Project/data/ra_body.pkl",
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
    "save_freq_model":5,
    "checkpoint": None,  # load pretrained weights
    "T_max": 50,  # cosine learning rate period (iteration)
    "eta_min": 0  # mininum learning rate
})

# device setting
config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % config.GPU_ID)
else:
    print('Using CPU')

results = np.load(config.SMPL_path)
print(results)

# data load
train_dataset = THREED_Dataset(
    path=config.db_path,
    train_mode=True,
    transforms=transforms.Compose([transforms.ToTensor()]),
    results=results
)

test_dataset = THREED_Dataset(
    path=config.db_path,
    train_mode=False,
    transforms=transforms.Compose([transforms.ToTensor()]),
    results=results
)
#
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                          drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         drop_last=True, shuffle=False)

# # save intermediate layers
# save_output = SaveOutput()
# hook_handles = []
# for layer in model_backbone.modules():
#     if isinstance(layer, Mixed_5b):
#         handle = layer.register_forward_hook(save_output)
#         hook_handles.append(handle)
#     elif isinstance(layer, Block35):
#         handle = layer.register_forward_hook(save_output)
#         hook_handles.append(handle)

# loss function
criterion = torch.nn.MSELoss()

# create model
model = UNet(n_channels=3, n_classes=3, bilinear=False).to(config.device)
model2 = UNet(n_channels=4, n_classes=3, bilinear=False).to(config.device)


# smpl_model = SMPL(constants.SMPL_MODEL_DIR).eval().to(config.device)
smpl_model = SMPL(model_path=constants.SMPL_MODEL_DIR, gender='NEUTRAL', batch_size=1).eval().to(config.device)


# pred_img_path_list = results['imgname']
# print(123123123123, pred_img_path_list)
# pred_betas = torch.from_numpy(results['shape']).float().to(config.device)
# pred_rotmat = torch.from_numpy(results['pose']).float().to(config.device)
# pred_cam_full = torch.from_numpy(results['global_t']).float().to(config.device)
# pred_joints = torch.from_numpy(results['pred_joints']).float().to(config.device)
# pred_focal_l = torch.from_numpy(results['focal_l']).float().to(config.device)
# pred_detection_all = torch.from_numpy(results['detection_all']).float().to(config.device)
# print('12378938472834938491849839823shape',pred_betas[0,:].shape)
# print('12378938472834938491849839823shape',pred_rotmat[0,:].shape)
# print('12378938472834938491849839823shape',pred_cam_full[0,:].shape)
# print('12378938472834938491849839823shape',pred_joints[0,:,:].shape)
# print('12378938472834938491849839823shape',pred_focal_l[0].shape)
# print('12378938472834938491849839823shape',pred_detection_all[0,:].shape)



optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
losses, scores = [], []
for epoch in range(start_epoch, config.n_epoch):
    loss = train_epoch(config, epoch, model, model2, criterion, optimizer, scheduler, train_loader, smpl_model)
    # print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

    if (epoch + 1) % config.val_freq == 0:
        loss = eval_epoch(config, epoch, model, model2, criterion, test_loader, smpl_model)
        # print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))
