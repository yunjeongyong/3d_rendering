import os
import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    TexturesUV,
)


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
    print('1231231312323123123',maxValue)
    print('123312313123231231',minValue)
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

def train_epoch(config, epoch, model1, model2, criterion, optimizer, scheduler, train_loader, smpl_model, raster):
    losses=[]
    model1.train()
    model2.train()
    with open(config.ra_body_path, 'rb') as f:
        this_dict = pickle.load(f)
    ra_faces_uv = this_dict['faces_uv'].to(config.device)
    ra_verts_uv = this_dict['verts_uv'].to(config.device)
    ra_faces = this_dict['faces'].to(config.device)

    for idx, data in tqdm(enumerate(train_loader)):
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


        pred_output = smpl_model(betas=betas,
                                 body_pose=rotmat[:, 1:],
                                 global_orient=rotmat[:, [0]],
                                 pose2rot=True,
                                 transl=cam_full)
        pred_vertices_t = pred_output.vertices
        pred_vertices_t = pred_vertices_t.to(config.device)
        faces_numpy = smpl_model.faces.astype(np.float)
        faces = torch.from_numpy(faces_numpy)
        print('img',img.shape)

        texture = model1(img).to(config.device)
        _, _, img_h, img_w = img.shape

        texture = texture.to(config.device)
        tex_map = torch.permute(texture.detach(), (0, 2, 3, 1))
        texture = torch.permute(texture, (0, 2, 3, 1))
        print('texture', texture.shape)
        print('texture', texture.shape)

        ra_faces_uv = ra_faces_uv.type_as(texture).long()
        textures = TexturesUV(maps=texture, faces_uvs=[ra_faces_uv for _ in range(batch)],
                              verts_uvs=[ra_verts_uv for _ in range(batch)]).to(config.device)

        smpl_mesh = Meshes(
            verts=[pred_vertices_t[0,:,:].to(config.device)],
            faces=[faces.to(config.device)],
            textures=textures.to(config.device)
        )

        # pytorch3d.io.save_obj('/media/mnt/Project/data/epoch%d_%d.obj'% (epoch + 1, idx), verts=pred_vertices_t[0,:,:], faces=faces, verts_uvs=ra_verts_uv, faces_uvs=ra_faces_uv, texture_map=texture.squeeze(0))
        fragments = raster(smpl_mesh)
        texels = smpl_mesh.sample_textures(fragments)
        out = texels[:, :, :, 0, :]
        out_ = out.permute(0, 3, 1, 2)
        print('out_shape',out_.shape)

        rgb_img = model2(out_)
        print('img.shape',img.shape)
        print('rgb_img', rgb_img.shape)
        loss_mse = criterion(rgb_img, img)
        loss_total = loss_mse
        loss_val = loss_total.item()
        losses.append(loss_val)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        print('[train] epoch:%d / loss:%f ' % (epoch + 1, loss_total.item()))

        # save weights
        if (epoch + 1) % config.save_freq == 0:
            img_file_name = "rgb_img_epoch%d.jpg" % (epoch + 1)
            img_save_file = os.path.join(config.save_img_path, img_file_name)
            pltshow(rgb_img, img_save_file)
            print('img_save_file', img_save_file)
            save_image(rgb_img, img_save_file)

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
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_total
            }, weights_file)
            print('save weights of epoch %d' % (epoch + 1))

        return np.mean(losses)


def eval_epoch(config, epoch, model1, model2, criterion, test_loader, smpl_model, raster):
    with torch.no_grad():
        losses = []
        model1.eval()
        model2.eval()

        with open(config.ra_body_path, 'rb') as f:
            this_dict = pickle.load(f)
        ra_faces_uv = this_dict['faces_uv'].to(config.device)
        ra_verts_uv = this_dict['verts_uv'].to(config.device)
        ra_faces = this_dict['faces'].to(config.device)

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

            pred_output = smpl_model(betas=betas,
                                     body_pose=rotmat[:, 1:],
                                     global_orient=rotmat[:, [0]],
                                     pose2rot=True,
                                     transl=cam_full)
            pred_vertices_t = pred_output.vertices
            pred_vertices_t = pred_vertices_t.to(config.device)
            faces_numpy = smpl_model.faces.astype(np.float)
            faces = torch.from_numpy(faces_numpy)
            print('img', img.shape)

            texture = model1(img).to(config.device)
            _, _, img_h, img_w = img.shape

            texture = texture.to(config.device)
            tex_map = torch.permute(texture.detach(), (0, 2, 3, 1))
            texture = torch.permute(texture, (0, 2, 3, 1))
            print('texture', texture.shape)
            print('texture', texture.shape)

            ra_faces_uv = ra_faces_uv.type_as(texture).long()
            textures = TexturesUV(maps=texture, faces_uvs=[ra_faces_uv for _ in range(batch)],
                                  verts_uvs=[ra_verts_uv for _ in range(batch)]).to(config.device)

            smpl_mesh = Meshes(
                verts=[pred_vertices_t[0, :, :].to(config.device)],
                faces=[faces.to(config.device)],
                textures=textures.to(config.device)
            )

            # pytorch3d.io.save_obj('/media/mnt/Project/data/epoch%d_%d.obj'% (epoch + 1, idx), verts=pred_vertices_t[0,:,:], faces=faces, verts_uvs=ra_verts_uv, faces_uvs=ra_faces_uv, texture_map=texture.squeeze(0))
            fragments = raster(smpl_mesh)
            texels = smpl_mesh.sample_textures(fragments)
            out = texels[:, :, :, 0, :]
            out_ = out.permute(0, 3, 1, 2)
            print('out_shape', out_.shape)

            rgb_img = model2(out_)
            print('img.shape', img.shape)
            print('rgb_img', rgb_img.shape)
            loss_mse = criterion(rgb_img, img)
            loss_total = loss_mse
            loss_val = loss_total.item()
            losses.append(loss_val)

        print('[test] epoch:%d / loss:%f ' % (epoch + 1, loss_total.item()))

        return np.mean(losses)













