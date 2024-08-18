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

def generate_uvmap(opts, img, seg, coord, tgt, model):
    src = torch.cat([img, seg], dim=1)  # Key

    tgt = tgt.expand(src.shape[0], -1, -1, -1)  # Query

    if not opts.mask_fusion:
        value = coord if opts.out_type == 'flow' else img
    else:
        value = torch.cat([coord, img], dim=1)
    out = model(tgt, src, value)

    # generate uvmap
    combine_mask = 0
    if not opts.mask_fusion:
        if opts.out_type == 'flow':
            uvmap = torch.nn.functional.grid_sample(img, out.permute(0, 2, 3, 1))
            # print('uvmap', uvmap.shape)

        elif opts.out_type == 'rgb':
            uvmap = out
            # print('uvmap', uvmap.shape)

    else:
        combine_mask = out[2]
        uvmap_flow = torch.nn.functional.grid_sample(img, out[0].permute(0, 2, 3, 1))
        uvmap_rgb = out[1]
        uvmap = uvmap_flow * combine_mask + uvmap_rgb * (1 - combine_mask)
        # print('uvmap', uvmap.shape)

    return uvmap, combine_mask


def train_epoch(config, opts, epoch, model1, model2, model3, criterion_m, criterion_p, optimizer, scheduler, train_loader, smpl_model, raster, tgt):
    losses=[]
    losses_d =[]
    model1.train()
    model2.train()
    model3.train()
    # modeld.train()

    verts, faces_dict, aux = load_obj('/media/mnt/Project/smpl_uv.obj')
    verts = verts.to(config.device) # (6890, 3)
    verts_uvs = aux.verts_uvs[None, ...].to(config.device)  # (1, 7576, 2)
    faces = faces_dict.verts_idx.to(config.device)  # (13766, 3)
    faces_uvs = faces_dict.textures_idx[None, ...].to(config.device)    # (1, 13766, 3)
    print('training start!')
    for idx, data in tqdm(enumerate(train_loader)):
        img = data['img'].to(config.device)
        coord = data['coord'].to(config.device)
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

        pred_output = smpl_model(betas=betas,
                                 body_pose=rotmat[:, 1:],
                                 global_orient=rotmat[:, [0]],
                                 pose2rot=True,
                                 transl=cam_full)

        rotmat2_shape = rotmat.shape
        rotmat2 = (torch.rand(rotmat2_shape) - 0.5) * 0.4
        body_pose2 = rotmat2[:, 1:].to(config.device)
        global_orient2 = rotmat2[:, [0]].to(config.device)

        betas2_shape = betas.shape
        betas2 = (torch.rand(betas2_shape) - 0.5) * 0.06
        betas2 = betas2.to(config.device)

        pred_output2 = smpl_model(betas=betas2,
                                  body_pose=body_pose2,
                                  global_orient=global_orient2,
                                  pose2rot=True,
                                  transl=cam_full)

        pred_verts = pred_output.vertices  # (1, 6890, 3)
        pred_verts = pred_verts.to(config.device)

        pred_verts2 = pred_output2.vertices
        pred_verts2 = pred_verts2.to(config.device)

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

        smpl_mesh2 = Meshes(
            verts=pred_verts2,
            faces=faces.unsqueeze(dim=0),
            textures=neural_texture_obj.to(config.device)
        )

        fragments = raster(smpl_mesh)
        fragments2 = raster(smpl_mesh2)
        texels = smpl_mesh.sample_textures(fragments)
        texels2 = smpl_mesh2.sample_textures(fragments2)
        rastered_out = texels[:, :, :, 0, :]  # (1, 64, 64, 8)
        rastered_out2 = texels2[:, :, :, 0, :]
        # print('rastered_out', rastered_out.shape)
        rastered_feature = rastered_out.permute(0, 3, 1, 2)  # (1, 8, 64, 64)
        rastered_feature2 = rastered_out2.permute(0, 3, 1, 2)

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
        # print('rastered_feature',rastered_feature.shape)
        rgb_recon = model2(rastered_feature)  # neural renderer
        rgb_recon2 = model2(rastered_feature2)

        uvmap, _ = generate_uvmap(opts, img, seg, coord, tgt, model3)
        uvmap2, _ = generate_uvmap(opts, rgb_recon, seg, coord, tgt, model3)

        loss_perceptual = criterion_p(input=rgb_recon, target=img, feature_layers=(0, 1, 2, 3), style_layers=())
        loss_perceptual = config.perceptual_weight * loss_perceptual
        loss_uvmap = criterion_m(uvmap2, uvmap)
        loss_mse = criterion_m(rgb_recon, img)
        loss_total = loss_uvmap + loss_mse + loss_perceptual
        loss_val = loss_total.item()
        losses.append(loss_val)

        optimizer.zero_grad()
        # loss_total.backward(retain_graph=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # out_d = modeld(img)
        # real_label = torch.ones(out_d.size()).to(config.device)
        # fake_label = torch.zeros(out_d.size()).to(config.device)
        # real_loss = criterion_gan(out_d, real_label)
        # fake_loss = criterion_gan(modeld(rgb_recon.detach()), fake_label)
        # loss_D = (real_loss + fake_loss) / 2
        #
        # optimizer_d.zero_grad()
        # loss_D.backward()
        # optimizer_d.step()

    print('[train] epoch:%d / loss:%f ' % (epoch + 1, loss_total.item()))

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
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_total,
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses)


def eval_epoch(config, opts, epoch, model1, model2, model3, criterion_m, criterion_p, test_loader, smpl_model, raster, tgt):
    with torch.no_grad():
        losses = []
        model1.eval()
        model2.eval()
        model3.eval()
        # modeld.eval()

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
        for idx, data in tqdm(enumerate(test_loader)):
            img = data['img'].to(config.device)
            coord = data['coord'].to(config.device)
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

            pred_output = smpl_model(betas=betas,
                                     body_pose=rotmat[:, 1:],
                                     global_orient=rotmat[:, [0]],
                                     pose2rot=True,
                                     transl=cam_full)

            rotmat2_shape = rotmat.shape
            rotmat2 = (torch.rand(rotmat2_shape) - 0.5) * 0.4
            body_pose2 = rotmat2[:, 1:].to(config.device)
            global_orient2 = rotmat2[:, [0]].to(config.device)

            betas2_shape = betas.shape
            betas2 = (torch.rand(betas2_shape) - 0.5) * 0.06
            betas2 = betas2.to(config.device)

            pred_output2 = smpl_model(betas=betas2,
                                      body_pose=body_pose2,
                                      global_orient=global_orient2,
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

            uvmap, _ = generate_uvmap(opts, img, seg, coord, tgt, model3)
            uvmap2, _ = generate_uvmap(opts, rgb_recon, seg, coord, tgt, model3)

            loss_perceptual = criterion_p(input=rgb_recon, target=img, feature_layers=(0, 1, 2, 3), style_layers=())
            loss_perceptual = config.perceptual_weight * loss_perceptual
            loss_uvmap = criterion_m(uvmap2, uvmap)
            loss_mse = criterion_m(rgb_recon, img)
            loss_total = loss_uvmap + loss_mse + loss_perceptual
            loss_val = loss_total.item()
            losses.append(loss_val)

            # out_d = modeld(img)
            # real_label = torch.ones(out_d.size()).to(config.device)
            # fake_label = torch.zeros(out_d.size()).to(config.device)
            # real_loss = criterion_gan(out_d, real_label)
            # fake_loss = criterion_gan(modeld(rgb_recon.detach()), fake_label)
            # loss_D = (real_loss + fake_loss) / 2


        print('[test] epoch:%d / loss:%f ' % (epoch + 1, loss_total.item()))

        return np.mean(losses)


