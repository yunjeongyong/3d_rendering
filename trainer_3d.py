import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
# from models.renderer import VGG_renderer
from models.decoder import Decoder
# from models.transformer import TransformerEncoderUnit
from options import TrainOptions
from models.texformer_qkv import Texformer_qkv
# from models.texformer import Texformer
from models.unet import UNet_
from models.discriminator import Discriminator, weights_init_normal
from models.replaybuffer import ReplayBuffer
from option.config import Config
from dataset_3d import THREED_Dataset
from train_3d import train_epoch, eval_epoch
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
        "snap_path": "/media/mnt/Project/data/rgb_img_perceptual_3d",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img_perceptual_3d",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "uv_encoding_path": "/media/mnt/Project/data/UV_color_encoding.npy",
        "train_size": 0.8,
        "scenes": "all",

        # ensemble in validation phase
        "lrD": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "channels":3,
        "img_height":256,
        "img_width":256,
        "img_h":128,
        "img_w":128,
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "perceptual_weight":0.5,
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
opts = TrainOptions().parse_args()
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
criterion_gan = torch.nn.MSELoss()


fake_B_buffer = ReplayBuffer()

# create model
model1 = UNet_(n_channels=3, n_classes=8).to(config.device)
# model2 = VGG_renderer().to(config.device)
model2 = Decoder(8, 64).to(config.device)
model3 = Texformer_qkv(opts).to(config.device)
# input_shape = (config.channels, config.img_height, config.img_width)
# modeld = Discriminator(input_shape).to(config.device)
# modeld.apply(weights_init_normal)

params = list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters())

# smpl_model = SMPL(constants.SMPL_MODEL_DIR).eval().to(config.device)
smpl_model = SMPL(model_path=constants.SMPL_MODEL_DIR, gender='NEUTRAL', batch_size=1).eval().to(config.device)

raster = rasterizer_setting(config, config.img_h, config.img_w)

optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer_d = torch.optim.Adam(modeld.parameters(), lr=config.lrD, betas=(config.beta1, config.beta2))
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

tgt = torch.from_numpy(np.load(config.uv_encoding_path)).permute(2, 0, 1)[None]
tgt = F.resize(tgt, 256)
tgt = tgt.to(config.device)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model2.load_state_dict(checkpoint['model_state_dict'])
    model3.load_state_dict(checkpoint['model_state_dict'])
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
    loss = train_epoch(config, opts, epoch, model1, model2, model3, criterion_m, criterion_p, optimizer, scheduler, train_loader, smpl_model, raster, tgt)

    if (epoch + 1) % config.val_freq == 0:
        loss = eval_epoch(config, opts, epoch, model1, model2, model3, criterion_m, criterion_p, test_loader, smpl_model, raster, tgt)
