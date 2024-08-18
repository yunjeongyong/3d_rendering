import os
import PIL.Image
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    TexturesUV,
)
from pytorch3d.io import load_obj
from srgan_utils import AverageMeter
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

def to_tanh(t):
    return t * 2 - 1.

def to_sigm(t):
    return (t + 1) / 2

def segment_img(img, segm):
    img = to_sigm(img) * segm
    img = to_tanh(img)
    return img

def create_cameras(
    R=None, T=None,
    azim=0, elev=0., dist=1.,
    fov=12., znear=0.01,
    device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras

def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def pltshow(img, save_path):
    img = img.detach()
    img_t = img.permute(0, 2, 3, 1)
    img_t = img_t.squeeze(0)
    img_t = img_t.cpu().numpy()
    maxValue = np.amax(img_t)
    minValue = np.amin(img_t)
    # img_t = img_t/np.amax(img_t)
    Image = np.clip(img_t, 0, 1)
    # Image \
    #     .fromarray((img_t * 255).astype(np.uint8)) \
    #     .save(save_path)
    plt.imshow(Image)
    return plt.show()

def torchtotensor(idx, img):
    img = torch.detach(img)
    img = torch.permute(img, (0, 2, 3, 1))  # 넘파이
    img = torch.clip(img, 0., 1.)
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    plt.imshow(img)
    plt.savefig('%s.png'% format(idx))
    return plt.show()

def torchimage(img):
    img = torch.detach(img)
    img = torch.clip(img, 0., 1.)
    img = img.squeeze(0)
    img = (img * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    plt.imshow(img)
    # img = img.cpu().numpy()
    # img = (img * 255).astype(np.uint8)
    return plt.show()


def torchtotensor2(idx, img):
    img = torch.permute(img, (0, 2, 3, 1))
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    plt.imshow(img)
    plt.savefig('%s_2.png'%format(idx))
    return plt.show()

def pilimgsave(img, save_file):
    img = img.cpu().detach()
    img = img.squeeze(0)
    if img.shape[0]==3:
        img = (np.transpose(img, (1,2,0)) + 1)/ 2.0 * 255.0
    elif img.shape[0] == 1:
        img = (img[0] +1) / 2.0 * 255.0
    np_arr = np.array(img, dtype=np.uint8)
    img = PIL.Image.fromarray(np_arr.astype(np.uint8))
    return img.save(save_file)

def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # ATTENTION: the line shoule be commented out as the points have been aligned
    # points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]



def train_epoch(config, epoch, model1, model_g, model_d, pixel_criterion, content_criterion, adversarial_criterion, optimizer_g, optimizer_d, scheduler_g, scheduler_d, train_loader, smpl_model, raster):
    g_losses=[]
    d_losses = []
    model1.train()
    model_g.train()
    model_d.train()

    # batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_recon_probabilities = AverageMeter("D(RGB_IMG)", ":6.3f")

    verts, faces_dict, aux = load_obj('/media/mnt/Project/smpl_uv.obj')
    verts = verts.to(config.device) # (6890, 3)
    verts_uvs = aux.verts_uvs[None, ...].to(config.device)  # (1, 7576, 2)
    faces = faces_dict.verts_idx.to(config.device)  # (13766, 3)
    faces_uvs = faces_dict.textures_idx[None, ...].to(config.device)    # (1, 13766, 3)
    print('training start!')
    for idx, data in tqdm(enumerate(train_loader)):
        img = data['img'].to(config.device)
        imgname = data['imgname']
        # print('imgname', imgname)
        seg = data['seg'].to(config.device)
        betas = data['betas'].to(config.device)
        rotmat = data['rotmat'].to(config.device)
        cam_full = data['cam_full'].to(config.device)
        joints = data['joints'].to(config.device)
        focal_l = data['focal_l'].to(config.device)
        detection_all = data['detection_all'].to(config.device)
        obj_file = data['obj_path']

        batch = img.shape[0]
        _, _, img_h, img_w = img.shape
        real_label = torch.full([batch, 1], 1.0, dtype=img.dtype, device=config.device)
        fake_label = torch.full([batch, 1], 0.0, dtype=img.dtype, device=config.device)

        pred_output = smpl_model(betas=betas,
                                 body_pose=rotmat[:, 1:],
                                 global_orient=rotmat[:, [0]],
                                 pose2rot=True,
                                 transl=cam_full)
        pred_verts = pred_output.vertices   # (1, 6890, 3)
        pred_verts = pred_verts.to(config.device)

        neural_texture = model1(img).to(config.device)
        neural_texture = neural_texture.to(config.device)
        neural_texture = torch.permute(neural_texture, (0, 2, 3, 1))  # (1, 512, 512, 8)

        neural_texture_obj = TexturesUV(maps=neural_texture, faces_uvs=faces_uvs.long(),
                              verts_uvs=verts_uvs.type_as(img))
        smpl_mesh = Meshes(
            verts=pred_verts,
            faces=faces.unsqueeze(dim=0),
            textures=neural_texture_obj.to(config.device)
        )

        fragments = raster(smpl_mesh)
        texels = smpl_mesh.sample_textures(fragments)
        rastered_out = texels[:, :, :, 0, :] # (1, 64, 64, 8)
        rastered_feature = rastered_out.permute(0, 3, 1, 2) # (1, 8, 64, 64)

        '''삭제할것! 안쓰는 부분 (rasterize 할 때 결과 이미지 추출 확인)'''
        # rasterrrrrrrr = rastered_feature.squeeze(0)
        # print('rasterrrrrrrr',rasterrrrrrrr.shape)
        # rasterrrrrrrr = rasterrrrrrrr[2, :, :]
        # rasterrrrrrrr = rasterrrrrrrr.detach().cpu().numpy()
        # plt.imshow(rasterrrrrrrr)
        # plt.show()
        '''삭제할것! 안쓰는 부분 (rasterize가 잘되는지 확인)'''
        # import cv2
        # from torchvision.transforms import ToTensor
        # to_tensor = ToTensor()
        # texture_sample = cv2.imread('/mnt/Projects/smpl_texture_sample.jpg')
        # texture_sample = cv2.resize(texture_sample, (512, 512))
        # texture_sample_t = to_tensor(texture_sample)
        # texture_sample_t = texture_sample_t.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # texture_sample = texture_sample_t.to(config.device)
        # textures_sample_obj = TexturesUV(maps=texture_sample, faces_uvs=faces_uvs.long().detach(),
        #                           verts_uvs=verts_uvs.type_as(img)).detach()  # 안씀 (시각화 용)
        #
        # smpl_mesh_sample = Meshes(
        #     verts=pred_verts.detach(),
        #     faces=faces.unsqueeze(dim=0).detach(),
        #     textures=textures_sample_obj.to(config.device)
        # )
        # fragments_sample = raster(smpl_mesh_sample)
        # texels_sample = smpl_mesh_sample.sample_textures(fragments_sample)
        # rastered_out_sample = texels_sample[:, :, :, 0, :]
        #
        # plt.figure(figsize=(7, 7))
        # texturesuv_image_matplotlib(smpl_mesh_sample.textures, subsample=None)
        # plt.axis("off")
        # plt.show()  # 시각화 용
        #
        # neural_texture_obj_tmp = TexturesUV(maps=neural_texture[..., 0:3], faces_uvs=faces_uvs.long(),
        #                                 verts_uvs=verts_uvs.type_as(img))
        # smpl_mesh_tmp = Meshes(
        #     verts=pred_verts.detach(),
        #     faces=faces.unsqueeze(dim=0).detach(),
        #     textures=neural_texture_obj_tmp.to(config.device)
        # )
        # fragments_tmp = raster(smpl_mesh_tmp)
        # texels_tmp = smpl_mesh_tmp.sample_textures(fragments_tmp)
        # rastered_out_tmp = texels_tmp[:, :, :, 0, :]
        #
        # plt.figure(figsize=(7, 7))
        # texturesuv_image_matplotlib(smpl_mesh_tmp.textures, subsample=None)
        # plt.axis("off")
        # plt.show()  # 시각화 용
        '''여기까지 삭제할것!'''
        for d_parameters in model_d.parameters():
            d_parameters.requires_grad = True

        model_d.zero_grad(set_to_none=True)
        gt_output = model_d(img)
        d_loss_gt = adversarial_criterion(gt_output, real_label)
        d_loss_gt.backward(retain_graph=True)

        print(rastered_feature.shape)
        rgb_recon = model_g(rastered_feature)
        print('rgb_recon', rgb_recon.shape)# neural renderer
        rgb_recon_output = model_d(rgb_recon.detach().clone())
        d_loss_recon = adversarial_criterion(rgb_recon_output, fake_label)
        d_loss_recon.backward()

        d_loss = d_loss_gt + d_loss_recon
        d_loss_val = d_loss.item()
        d_losses.append(d_loss_val)

        optimizer_d.step()

        for d_parameters in model_d.parameters():
            d_parameters.requires_grad = False

        model_g.zero_grad(set_to_none=True)

        pixel_loss = config.pixel_weight * pixel_criterion(rgb_recon, img)
        content_loss = config.content_weight * content_criterion(rgb_recon, img)
        adversarial_loss = config.adversarial_weight * adversarial_criterion(model_d(rgb_recon), real_label)

        g_loss = pixel_loss + content_loss + adversarial_loss
        g_loss_val = g_loss.item()
        g_losses.append(g_loss_val)
        g_loss.backward()
        optimizer_g.step()

        d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))
        d_recon_probability = torch.sigmoid_(torch.mean(rgb_recon_output.detach()))

        pixel_losses.update(pixel_loss.item(), rastered_feature.size(0))
        content_losses.update(content_loss.item(), rastered_feature.size(0))
        adversarial_losses.update(adversarial_loss.item(), rastered_feature.size(0))
        d_gt_probabilities.update(d_gt_probability.item(), rastered_feature.size(0))
        d_recon_probabilities.update(d_recon_probability.item(), rastered_feature.size(0))

        # batch_time.update(time.time() - end)
        # end = time.time()
        # loss_mse = criterion(rgb_recon, img)
        # loss_total = loss_mse
        # loss_val = loss_total.item()
        # losses.append(loss_val)
        scheduler_d.step()
        scheduler_g.step()

    print('[train] epoch:%d / Train_D_loss:%f ' % (epoch + 1, d_loss.item()))
    print('[train] epoch:%d / Train_G_loss:%f ' % (epoch + 1, g_loss.item()))
    print('[train] epoch:%d / Train_pixel_loss:%f ' % (epoch + 1, pixel_loss.item()))
    print('[train] epoch:%d / Train_content_loss:%f ' % (epoch + 1, content_loss.item()))
    print('[train] epoch:%d / Train_adversarial_loss:%f ' % (epoch + 1, adversarial_loss.item()))
    print('[train] epoch:%d / Train_d_probability:%f ' % (epoch + 1, d_gt_probability.item()))
    print('[train] epoch:%d / Train_recon_probability:%f ' % (epoch + 1, d_recon_probability.item()))
    # save weights
    if (epoch + 1) % config.save_freq == 0:
        img_file_name = "rgb_recon_img_epoch%d.jpg" % (epoch + 1)
        img_save_file = os.path.join(config.save_img_path, img_file_name)
        # # pltshow(rgb_recon, img_save_file)
        # print('img_save_file', img_save_file)
        save_image(rgb_recon, img_save_file)

        gt_file_name = "gt_img_epoch%d.jpg" % (epoch + 1)
        gt_save_file = os.path.join(config.save_img_path, gt_file_name)
        pltshow(img, gt_save_file)
        save_image(img, gt_save_file)
    if (epoch + 1) % config.save_freq_model == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)

        torch.save({
            'epoch': epoch,
            'model_encoder_state_dict': model1.state_dict(),
            'model_g_state_dict': model_g.state_dict(),
            'model_d_state_dict': model_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'loss_d': d_loss,
            'loss_g': g_loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(d_losses), np.mean(g_losses)


def eval_epoch(config, epoch, model1, model_g, model_d, pixel_criterion, content_criterion, adversarial_criterion, test_loader, smpl_model, raster):
    with torch.no_grad():
        d_losses = []
        g_losses = []
        model1.eval()
        model_g.eval()
        model_d.eval()

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        pixel_losses = AverageMeter("Pixel loss", ":6.6f")
        content_losses = AverageMeter("Content loss", ":6.6f")
        adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
        d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
        d_recon_probabilities = AverageMeter("D(RGB_IMG)", ":6.3f")

        verts, faces_dict, aux = load_obj('/media/mnt/Project/smpl_uv.obj')
        verts = verts.to(config.device)  # (6890, 3)
        verts_uvs = aux.verts_uvs[None, ...].to(config.device)  # (1, 7576, 2)
        faces = faces_dict.verts_idx.to(config.device)  # (13766, 3)
        faces_uvs = faces_dict.textures_idx[None, ...].to(config.device)  # (1, 13766, 3)
        print('test starting!')

        for idx, data in tqdm(enumerate(test_loader)):
            img = data['img'].to(config.device)
            seg = data['seg'].to(config.device)
            betas = data['betas'].to(config.device)
            rotmat = data['rotmat'].to(config.device)
            cam_full = data['cam_full'].to(config.device)
            joints = data['joints'].to(config.device)
            focal_l = data['focal_l'].to(config.device)
            detection_all = data['detection_all'].to(config.device)
            obj_file = data['obj_path']

            batch = img.shape[0]
            _, _, img_h, img_w = img.shape
            real_label = torch.full([batch, 1], 1.0, dtype=img.dtype, device=config.device)
            fake_label = torch.full([batch, 1], 0.0, dtype=img.dtype, device=config.device)

            pred_output = smpl_model(betas=betas,
                                     body_pose=rotmat[:, 1:],
                                     global_orient=rotmat[:, [0]],
                                     pose2rot=True,
                                     transl=cam_full)
            pred_verts = pred_output.vertices  # (1, 6890, 3)
            pred_verts = pred_verts.to(config.device)

            neural_texture = model1(img).to(config.device)  # (1, 8, 512, 512)
            neural_texture = neural_texture.to(config.device)
            neural_texture = torch.permute(neural_texture, (0, 2, 3, 1))  # (1, 512, 512, 8)

            neural_texture_uv = TexturesUV(maps=neural_texture, faces_uvs=faces_uvs.long(),
                                           verts_uvs=verts_uvs.type_as(img))
            smpl_mesh = Meshes(
                verts=pred_verts,
                faces=faces.unsqueeze(dim=0),
                textures=neural_texture_uv.to(config.device)
            )

            fragments = raster(smpl_mesh)
            texels = smpl_mesh.sample_textures(fragments)
            rastered_feature = texels[:, :, :, 0, :] #
            rastered_out = rastered_feature.permute(0, 3, 1, 2)

            gt_output = model_d(img)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

            rgb_recon = model_g(rastered_out)  # neural renderer
            rgb_recon_output = model_d(rgb_recon.detach().clone())
            d_loss_recon = adversarial_criterion(rgb_recon_output, fake_label)

            d_loss = d_loss_gt + d_loss_recon
            d_loss_val = d_loss.item()
            d_losses.append(d_loss_val)


            pixel_loss = config.pixel_weight * pixel_criterion(rgb_recon, img)
            content_loss = config.content_weight * content_criterion(rgb_recon, img)
            adversarial_loss = config.adversarial_weight * adversarial_criterion(model_d(rgb_recon), real_label)

            g_loss = pixel_loss + content_loss + adversarial_loss
            g_loss_val = g_loss.item()
            g_losses.append(g_loss_val)

            d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))
            d_recon_probability = torch.sigmoid_(torch.mean(rgb_recon_output.detach()))

            pixel_losses.update(pixel_loss.item(), rastered_out.size(0))
            content_losses.update(content_loss.item(), rastered_out.size(0))
            adversarial_losses.update(adversarial_loss.item(), rastered_out.size(0))
            d_gt_probabilities.update(d_gt_probability.item(), rastered_out.size(0))
            d_recon_probabilities.update(d_recon_probability.item(), rastered_out.size(0))

            # batch_time.update(time.time() - end)
            # end = time.time()

            # rgb_recon = model2(rastered_out)
            #
            # loss_mse = criterion(rgb_recon, img)
            # loss_total = loss_mse
            # loss_val = loss_total.item()
            # losses.append(loss_val)

        print('[test] epoch:%d / test_D_loss:%f ' % (epoch + 1, d_loss.item()))
        print('[test] epoch:%d / test_G_loss:%f ' % (epoch + 1, g_loss.item()))
        print('[test] epoch:%d / test_pixel_loss:%f ' % (epoch + 1, pixel_loss.item()))
        print('[test] epoch:%d / test_content_loss:%f ' % (epoch + 1, content_loss.item()))
        print('[test] epoch:%d / test_adversarial_loss:%f ' % (epoch + 1, adversarial_loss.item()))
        print('[test] epoch:%d / test_d_probability:%f ' % (epoch + 1, d_gt_probability.item()))
        print('[test] epoch:%d / test_recon_probability:%f ' % (epoch + 1, d_recon_probability.item()))

        return np.mean(d_losses), np.mean(g_losses)