import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# from models.renderer import VGG_renderer
from models.decoder import Decoder
# from models.transformer import TransformerEncoderUnit
from options import TrainOptions
from models.texformer import Texformer
from models.discriminator import Discriminator, weights_init_normal
from option.config import Config
from dataset_unettrans import THREED_Dataset
from train_unettrans_gan import train_epoch, eval_epoch
from models.smpl import SMPL
from common import constants
import numpy as np
from vgg_perceptual_loss import VGGPerceptualLoss
from util import rasterizer_setting


# config file
config = Config({
        # device
        "GPU_ID": "1",
        "num_workers": 2,

        # data
        "db_path": "/media/mnt/dataset",
        "seg_path": "/media/mnt/dataset/DeepFashion_segm_black_512padding/",
        # "SMPL_path":"/media/mnt/Project/data/i_cliff_hr48.npz",
        "SMPL_path": "/media/mnt/dataset/DeepFashion_image_black_512pad_cliff_deepfashion_hr48.npz",
        "snap_path": "/media/mnt/Project/data/rgb_img_7_5_transgan",  # path for saving weights
        "save_img_path": "/media/mnt/Project/data/rgb_img_7_5_transgan",
        "ra_body_path": "/media/mnt/Project/data/ra_body.pkl",
        "train_size": 0.8,
        "scenes": "all",

        # ensemble in validation phase
        "img_h":128,
        "img_w":128,
        "gan_img_h":256,
        "gan_img_w":256,
        "lambda_pixel":100,
        "test_ensemble": True,
        "n_ensemble": 5,
        # learning rate 빼기, encoder만
        # optimization
        "batch_size": 1,
        "perceptual_weight":0.5,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "start_epoch": 0,
        "n_epoch": 800,
        "val_freq": 1,
        "save_freq": 1,
        "save_freq_model": 20,
        # "checkpoint": "/media/mnt/Project/model_resume/epoch500.pth",  # load pretrained weights
        "checkpoint":None,
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
                         drop_last=True, shuffle=True)


# loss function
# feature_model_extractor_node = "features.35"
# feature_model_normalize_mean = [0.485, 0.456, 0.406]
# feature_model_normalize_std = [0.229, 0.224, 0.225]
criterion_m = torch.nn.L1Loss()
criterion_p = VGGPerceptualLoss(resize=False)
criterion_gan = torch.nn.L1Loss()
# criterion_p = srgan.content_loss(feature_model_extractor_node=feature_model_extractor_node, feature_model_normalize_mean=feature_model_normalize_mean, feature_model_normalize_std=feature_model_normalize_std)
criterion_m = criterion_m.to(config.device)
criterion_p = criterion_p.to(config.device)
criterion_gan = criterion_gan.to(config.device)

# Calculate output of image discriminator (PatchGAN)
patch = (1, config.gan_img_h // 2 ** 4, config.gan_img_w // 2 ** 4)
# create model
model1 = Texformer(opts).to(config.device)
# model2 = VGG_renderer().to(config.device)
model2 = Decoder(8, 64).to(config.device)
model3 = Discriminator().to(config.device)

if config.start_epoch != 0:
    # Load pretrained models
    model3.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (config.dataset_name, config.start_epoch)))
else:
    # Initialize weights
    model3.apply(weights_init_normal)

params = list(model1.parameters()) + list(model2.parameters())

# smpl_model = SMPL(constants.SMPL_MODEL_DIR).eval().to(config.device)
smpl_model = SMPL(model_path=constants.SMPL_MODEL_DIR, gender='NEUTRAL', batch_size=1).eval().to(config.device)

raster = rasterizer_setting(config, config.img_h, config.img_w)

optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
optimizer_D = torch.optim.Adam(model3.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])
    model3.load_state_dict(checkpoint['model3_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    loss_D = checkpoint['loss_D']

else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
losses, scores = [], []
for epoch in range(start_epoch, config.n_epoch):
    print('start_epoch',start_epoch)
    loss = train_epoch(config, epoch, patch, model1, model2, model3, criterion_m, criterion_p, criterion_gan, optimizer, optimizer_D, scheduler, scheduler_D, train_loader, smpl_model, raster)

    if (epoch + 1) % config.val_freq == 0:
        loss = eval_epoch(config, epoch, patch, model1, model2, model3, criterion_m, criterion_p, criterion_gan, test_loader, smpl_model, raster)
