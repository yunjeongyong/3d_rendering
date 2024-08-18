import os
import PIL.Image
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

def train_epoch(config, epoch, patch, model1, model2, model3, criterion_m, criterion_p, criterion_gan, optimizer, optimizer_D, scheduler, scheduler_D, train_loader, smpl_model, raster):
    losses=[]
    model1.train()
    model2.train()
    print('epoch_num', epoch)

    betas_2 = None
    body_pose_2 = None
    global_orient_2 = None

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

        valid = torch.ones((img.size(0), *patch), requires_grad=False)
        fake = torch.zeros((img.size(0), *patch), requires_grad=False)
        valid = valid.to(config.device)
        fake = fake.to(config.device)

        batch = img.shape[0]
        _, _, img_h, img_w = img.shape

        if idx == 0:
            betas_2 = betas.clone()
            body_pose_2 = rotmat[:, 1:].clone()
            global_orient_2 = rotmat[:, [0]].clone()

        pred_output = smpl_model(betas=betas,
                                 body_pose=rotmat[:, 1:],
                                 global_orient=rotmat[:, [0]],
                                 pose2rot=True,
                                 transl=cam_full)

        print("betas_2", betas_2)
        print("body_pose_2", body_pose_2)
        print("global_orient_2", global_orient_2)

        print("betas", betas)
        print("body_pose", rotmat[:, 1:])
        print("global_orient", rotmat[:, [0]])

        # rotmat2_shape = rotmat.shape
        # rotmat2 = (torch.rand(rotmat2_shape) - 0.5) * 0.4
        # body_pose2 = rotmat2[:, 1:].to(config.device)
        # global_orient2 = rotmat2[:, [0]].to(config.device)

        # betas2_shape = betas.shape
        # betas2 = (torch.rand(betas2_shape) - 0.5) * 0.06
        # betas2 = betas2.to(config.device)
        # betas2_shape = betas.shape
        # betas2 = betas2_shape
        # betas2 = betas2.to(config.device)

        pred_output2 = smpl_model(betas=betas_2,
                                  body_pose=body_pose_2,
                                  global_orient=global_orient_2,
                                  pose2rot=True,
                                  transl=cam_full)

        pred_verts = pred_output.vertices  # (1, 6890, 3)
        pred_verts = pred_verts.to(config.device)

        pred_verts2 = pred_output2.vertices
        pred_verts2 = pred_verts2.to(config.device)

        neural_texture = model1(img).to(config.device)  # (1, 8, 512, 512)
        neural_texture = neural_texture.to(config.device)
        neural_texture = torch.permute(neural_texture, (0, 2, 3, 1))  # (1, 512, 512, 8)
        neural_texture_uv = TexturesUV(maps=neural_texture, faces_uvs=faces_uvs.long(), verts_uvs=verts_uvs.type_as(img))
        smpl_mesh = Meshes(
            verts=pred_verts,
            faces=faces.unsqueeze(dim=0),
            textures=neural_texture_uv.to(config.device)
        )

        smpl_mesh2 = Meshes(
            verts=pred_verts2,
            faces=faces.unsqueeze(dim=0),
            textures=neural_texture_uv.to(config.device)
        )

        fragments = raster(smpl_mesh)
        fragments2 = raster(smpl_mesh2)
        texels = smpl_mesh.sample_textures(fragments)
        texels2 = smpl_mesh2.sample_textures(fragments2)
        rastered_feature = texels[:, :, :, 0, :]  # (1, 64, 64, 8)
        rastered_feature2 = texels2[:, :, :, 0, :]

        rastered_out = rastered_feature.permute(0, 3, 1, 2)  # (1, 8, 64, 64)
        rastered_out2 = rastered_feature2.permute(0, 3, 1, 2)

        rgb_recon = model2(rastered_out)
        rgb_recon2 = model2(rastered_out2)
        pred_fake = model3(rgb_recon, img)

        # print('rgb_recon', rgb_recon.shape)
        # print('img', img.shape)
        loss_gan = criterion_gan(pred_fake, valid)
        loss_perceptual = criterion_p(input=rgb_recon, target=img, feature_layers=(0, 1, 2, 3), style_layers=())
        loss_perceptual = config.perceptual_weight * loss_perceptual
        loss_mse = criterion_m(rgb_recon, img)
        loss_total = loss_gan + config.lambda_pixel * loss_mse + loss_perceptual
        loss_val = loss_total.item()
        losses.append(loss_val)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        optimizer_D.zero_grad()

        pred_real = model3(img, img)
        loss_real = criterion_gan(pred_real, valid)

        pred_fake = model3(rgb_recon.detach(), img)
        loss_fake = criterion_gan(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()
        scheduler_D.step()

    print('[train] epoch:%d / loss:%f / d_loss:%f ' % (epoch + 1, loss_total.item(), loss_D.item()))

    # save weights
    if (epoch + 1) % config.save_freq == 0:
        img_file_name = "rgb_img_epoch%d.jpg" % (epoch + 1)
        img_file_name2 = "rgb_img_pose2%d.jpg" % (epoch + 1)
        img_save_file = os.path.join(config.save_img_path, img_file_name)
        img_save_file2 = os.path.join(config.save_img_path, img_file_name2)
        pltshow(rgb_recon, img_save_file)
        print('img_save_file', img_save_file)
        save_image(rgb_recon, img_save_file)
        save_image(rgb_recon2, img_save_file2)

        gt_file_name = "gt_img_epoch%d.jpg" % (epoch + 1)
        gt_save_file = os.path.join(config.save_img_path, gt_file_name)
        pltshow(img, gt_save_file)
        save_image(img, gt_save_file)
    if (epoch + 1) % config.save_freq_model == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)

        torch.save({
            'epoch': epoch,
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'model3_state_dict': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'loss': loss_total,
            'loss_D': loss_D
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses)


def eval_epoch(config, epoch, patch, model1, model2, model3, criterion_m, criterion_p, criterion_gan, test_loader, smpl_model, raster):
    with torch.no_grad():
        losses = []
        model1.eval()
        model2.eval()
        print('test_epoch_num',epoch)
        # batch_size = config.batch_size
        # verts, faces_dict, aux = load_obj('/media/mnt/Project/smpl_uv.obj')
        # verts = verts.to(config.device)  # (6890, 3)
        # verts_uvs_ = aux.verts_uvs.to(config.device)  # (7576, 2)
        # verts_uvs = verts_uvs_.expand(batch_size, -1, -1)
        # print('verts_uvs', verts_uvs.shape)
        # faces_ = faces_dict.verts_idx.to(config.device)  # (13766, 3)
        # faces = faces_.expand(batch_size, -1, -1)
        # faces_uvs_ = faces_dict.textures_idx.to(config.device)  # (1, 13766, 3)
        # faces_uvs = faces_uvs_.expand(batch_size, -1, -1)
        #
        verts, faces_dict, aux = load_obj('/media/mnt/Project/smpl_uv.obj')
        verts = verts.to(config.device)  # (6890, 3)
        verts_uvs = aux.verts_uvs[None, ...].to(config.device)  # (1, 7576, 2)
        faces = faces_dict.verts_idx.to(config.device)  # (13766, 3)
        faces_uvs = faces_dict.textures_idx[None, ...].to(config.device)  # (1, 13766, 3)
        print('test starting!')

        betas_2 = None
        body_pose_2 = None
        global_orient_2 = None

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
            valid = torch.ones((img.size(0), *patch), requires_grad=False)
            fake = torch.zeros((img.size(0), *patch), requires_grad=False)
            valid = valid.to(config.device)
            fake = fake.to(config.device)


            batch = img.shape[0]
            _, _, img_h, img_w = img.shape

            if idx == 0:
                betas_2 = betas.clone()
                body_pose_2 = rotmat[:, 1:].clone()
                global_orient_2 = rotmat[:, [0]].clone()


            pred_output = smpl_model(betas=betas,
                                     body_pose=rotmat[:, 1:],
                                     global_orient=rotmat[:, [0]],
                                     pose2rot=True,
                                     transl=cam_full)
            # rotmat2_shape = rotmat.shape
            # rotmat2 = (torch.rand(rotmat2_shape) - 0.5) * 0.4
            # body_pose2 = rotmat2[:, 1:].to(config.device)
            # global_orient2 = rotmat2[:, [0]].to(config.device)
            #
            # betas2_shape = betas.shape
            # betas2 = (torch.rand(betas2_shape) - 0.5) * 0.06
            # betas2 = betas2.to(config.device)

            pred_output2 = smpl_model(betas=betas_2,
                                      body_pose=body_pose_2,
                                      global_orient=global_orient_2,
                                      pose2rot=True,
                                      transl=cam_full)

            pred_verts = pred_output.vertices  # (1, 6890, 3)
            pred_verts = pred_verts.to(config.device)

            pred_verts2 = pred_output2.vertices
            pred_verts2 = pred_verts2.to(config.device)

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

            smpl_mesh2 = Meshes(
                verts=pred_verts2,
                faces=faces.unsqueeze(dim=0),
                textures=neural_texture_uv.to(config.device)
            )

            fragments = raster(smpl_mesh)
            fragments2 = raster(smpl_mesh2)
            texels = smpl_mesh.sample_textures(fragments)
            texels2 = smpl_mesh2.sample_textures(fragments2)
            rastered_feature = texels[:, :, :, 0, :]  # (1, 64, 64, 8)
            rastered_feature2 = texels2[:, :, :, 0, :]

            rastered_out = rastered_feature.permute(0, 3, 1, 2)  # (1, 8, 64, 64)
            rastered_out2 = rastered_feature2.permute(0, 3, 1, 2)

            rgb_recon = model2(rastered_out)
            rgb_recon2 = model2(rastered_out2)
            pred_fake = model3(rgb_recon, img)

            # print('rgb_recon', rgb_recon.shape)
            # print('img', img.shape)
            loss_gan = criterion_gan(pred_fake, valid)
            loss_perceptual = criterion_p(input=rgb_recon, target=img, feature_layers=(0, 1, 2, 3), style_layers=())
            loss_perceptual = config.perceptual_weight * loss_perceptual
            loss_mse = criterion_m(rgb_recon, img)
            loss_total = loss_gan + config.lambda_pixel * loss_mse + loss_perceptual
            loss_val = loss_total.item()
            losses.append(loss_val)

            pred_real = model3(img, img)
            loss_real = criterion_gan(pred_real, valid)

            pred_fake = model3(rgb_recon.detach(), img)
            loss_fake = criterion_gan(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)


        print('[test] epoch:%d / loss:%f / loss_D:%f' % (epoch + 1, loss_total.item(), loss_D.item()))

        if (epoch + 1) % config.save_freq == 0:
            test_img_file_name = "test_rgb_img_epoch%d.jpg" % (epoch + 1)
            test_img_file_name2 = "test_rgb_img_pose2%d.jpg" % (epoch + 1)
            test_img_save_file = os.path.join(config.save_img_path, test_img_file_name)
            test_img_save_file2 = os.path.join(config.save_img_path, test_img_file_name2)
            pltshow(rgb_recon, test_img_save_file)
            print('test_img_save_file', test_img_save_file)
            print('test_img_save_file2', test_img_save_file2)

            save_image(rgb_recon, test_img_save_file)
            save_image(rgb_recon2, test_img_save_file2)

            test_gt_file_name = "test_gt_img_epoch%d.jpg" % (epoch + 1)
            test_gt_save_file = os.path.join(config.save_img_path, test_gt_file_name)
            pltshow(img, test_gt_save_file)
            save_image(img, test_gt_save_file)

        return np.mean(losses)