import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# from models.renderer import VGG_renderer
# from models.decoder import Decoder
import models.srgan as srgan
from models.srgan import SRResNet
from models.srgan import discriminator
# from models.transformer import TransformerEncoderUnit
from models.vgg import VGG
from models.unet import UNet_
from option.config import Config
from dataset import THREED_Dataset
from train_srgan import train_epoch, eval_epoch
from models.smpl import SMPL
from common import constants
import numpy as np
from utils import rasterizer_setting


# config file
config = Config({
        # device
        "GPU_ID": "2",
        "num_workers": 2,

        # data
        "db_path": "/media/mnt/dataset",
        "seg_path": "/media/mnt/dataset/DeepFashion_segm_black_512padding/",
        # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
        "SMPL_path": "/media/mnt/dataset/DeepFashion_image_black_512pad_cliff_deepfashion_hr48.npz",
        "snap_path": "/media/mnt/Project/data/rgb_img_gan_128",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img_gan_128",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "train_size": 0.8,
        "scenes": "all",


        "pixel_weight":1.0,
        "content_weight":1.0,
        "adversarial_weight":0.001,
        # ensemble in validation phase
        "img_h":128,
        "img_w":128,
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "n_epoch": 300,
        "val_freq": 1,
        "save_freq": 1,
        "save_freq_model": 10,
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
#
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                          drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         drop_last=True, shuffle=False)


# create model
model1 = VGG().to(config.device)
# model1 = TransformerEncoderUnit(feat_dim=64, n_head=4, pos_en_flag=True, attn_type='softmax').to(config.device)
# model2 = VGG_renderer().to(config.device)
model_g = SRResNet(in_channels=3, out_channels=3, channels=64, num_rcb=16, upscale_factor=4).to(config.device)
model_d = discriminator().to(config.device)


# loss function
feature_model_extractor_node = "features.35"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]
pixel_criterion = torch.nn.MSELoss()
content_criterion = srgan.content_loss(feature_model_extractor_node=feature_model_extractor_node, feature_model_normalize_mean=feature_model_normalize_mean, feature_model_normalize_std=feature_model_normalize_std)
adversarial_criterion = torch.nn.BCEWithLogitsLoss()
pixel_criterion = pixel_criterion.to(config.device)
content_criterion = content_criterion.to(config.device)
adversarial_criterion = adversarial_criterion.to(config.device)


# smpl_model = SMPL(constants.SMPL_MODEL_DIR).eval().to(config.device)
smpl_model = SMPL(model_path=constants.SMPL_MODEL_DIR, gender='NEUTRAL', batch_size=1).eval().to(config.device)

ras_sett = rasterizer_setting(config, config.img_h, config.img_w)


params_g = list(model1.parameters()) + list(model_g.parameters())
params_d = model_d.parameters()
optimizer_g = torch.optim.Adam(params_g, lr=config.learning_rate, weight_decay=config.weight_decay)
optimizer_d = torch.optim.Adam(params_d, lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config.T_max, eta_min=config.eta_min)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config.T_max, eta_min=config.eta_min)
# load weights & optimizer

if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model_g.load_state_dict(checkpoint['model_state_dict'])
    model_d.load_state_dict(checkpoint['model_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_state_dict'])
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
    dloss, gloss = train_epoch(config, epoch, model1, model_g, model_d, pixel_criterion, content_criterion, adversarial_criterion, optimizer_g, optimizer_d, scheduler_g, scheduler_d, train_loader, smpl_model, ras_sett)

    if (epoch + 1) % config.val_freq == 0:
        dloss, gloss = eval_epoch(config, epoch, model1, model_g, model_d, pixel_criterion, content_criterion, adversarial_criterion, test_loader, smpl_model, ras_sett)
