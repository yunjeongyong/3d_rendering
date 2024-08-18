import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# from models.renderer import VGG_renderer
from models.decoder import Decoder
# from models.transformer import TransformerEncoderUnit
from models.unet import UNet_
import models.srgan as srgan
from option.config import Config
from dataset import THREED_Dataset
from train_prof import train_epoch, eval_epoch
from models.smpl import SMPL
from common import constants
import numpy as np
from vgg_perceptual_loss import VGGPerceptualLoss
from utils import rasterizer_setting


# config file
config = Config({
        # device
        "GPU_ID": "0",
        "num_workers": 2,

        # data
        "db_path": "/media/mnt/dataset",
        "seg_path": "/media/mnt/dataset/DeepFashion_segm_black_512padding/",
        # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
        "SMPL_path": "/media/mnt/dataset/DeepFashion_image_black_512pad_cliff_deepfashion_hr48.npz",
        "snap_path": "/media/mnt/Project/data/rgb_img_perceptual5_weight1",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img_perceptual5_weight1",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "train_size": 0.8,
        "scenes": "all",

        # ensemble in validation phase
        "img_h":128,
        "img_w":128,
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "perceptual_weight":1,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
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


# loss function
# feature_model_extractor_node = "features.35"
# feature_model_normalize_mean = [0.485, 0.456, 0.406]
# feature_model_normalize_std = [0.229, 0.224, 0.225]
criterion_m = torch.nn.L1Loss()
criterion_p = VGGPerceptualLoss(resize=False)
# criterion_p = srgan.content_loss(feature_model_extractor_node=feature_model_extractor_node, feature_model_normalize_mean=feature_model_normalize_mean, feature_model_normalize_std=feature_model_normalize_std)
criterion_m = criterion_m.to(config.device)
criterion_p = criterion_p.to(config.device)


# create model
model1 = UNet_(n_channels=3, n_classes=8).to(config.device)
# model2 = VGG_renderer().to(config.device)
model2 = Decoder(8, 64).to(config.device)

params = list(model1.parameters()) + list(model2.parameters())

# smpl_model = SMPL(constants.SMPL_MODEL_DIR).eval().to(config.device)
smpl_model = SMPL(model_path=constants.SMPL_MODEL_DIR, gender='NEUTRAL', batch_size=1).eval().to(config.device)

ras_sett = rasterizer_setting(config, config.img_h, config.img_w)

optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model2.load_state_dict(checkpoint['model_state_dict'])
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
    loss = train_epoch(config, epoch, model1, model2, criterion_m, criterion_p, optimizer, scheduler, train_loader, smpl_model, ras_sett)

    if (epoch + 1) % config.val_freq == 0:
        loss = eval_epoch(config, epoch, model1, model2, criterion_m, criterion_p, test_loader, smpl_model, ras_sett)
