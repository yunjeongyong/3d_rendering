import os

import pytorch3d.io
import torch
import torch.nn.functional as nn
from tqdm import tqdm
import numpy as np
from torch.nn.parameter import Parameter
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle
from models.smpl import SMPL
from common import constants
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    Textures,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    AmbientLights,
    TexturesUV,
    TexturesVertex
)


def compute_k(focal_length, principal_point, height=512, width=512):
    fx = focal_length[0]
    fy = focal_length[1]
    px = principal_point[0]
    py = principal_point[1]

    K = np.zeros((4, 4))
    K[0][0] = 2 * fx / width
    K[1][1] = 2 * fy / height
    K[0][2] = 1.0 - 2.0 * px / height
    K[1][2] = 2.0 * py / height - 1.0
    K[3][2] = -1.0
    n = 0.05
    f = 100.0
    K[2][2] = (f + n) / (n - f)
    K[2][3] = (2 * f * n) / (n - f)
    return K

def torchtotensor(idx, img):
    img = torch.permute(img, (0, 2, 3, 1))
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    plt.imshow(img)
    plt.savefig('%s.png'% format(idx))
    return plt.show()

def torchtotensor2(idx, img):
    img = torch.permute(img, (0, 2, 3, 1))
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    plt.imshow(img)
    plt.savefig('%s_2.png'%format(idx))
    return plt.show()



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

def train_epoch(config, epoch, model, model2, criterion, optimizer, scheduler, train_loader, smpl_model):
    losses=[]
    model.train()
    with open(config.ra_body_path, 'rb') as f:
        this_dict = pickle.load(f)
    ra_faces_uv = this_dict['faces_uv'].to(config.device)
    ra_verts_uv = this_dict['verts_uv'].to(config.device)
    ra_faces = this_dict['faces'].to(config.device)

    for idx, data in tqdm(enumerate(train_loader)):
        img = data['img'].to(config.device)
        betas = data['betas'].to(config.device)
        rotmat = data['rotmat'].to(config.device)
        cam_full =data['cam_full'].to(config.device)
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
        # pred_vertices_np = pred_vertices.cpu().numpy()
        faces_numpy = smpl_model.faces.astype(np.float)
        faces = torch.from_numpy(faces_numpy)
        faces = torch.tensor(faces, dtype=torch.int64).to(config.device)
        print('img',img.shape)

        verts, face, aux = load_obj(obj_file[0])
        verts = verts.to(config.device)
        verts = torch.tensor(verts, dtype=torch.float)

        # _, _, img_h, img_w = img.shape
        # print(img_h)
        #
        # img_h = torch.tensor([img.shape[2]]).to(config.device)
        # img_w = torch.tensor([img.shape[3]])
        # print(img_h.shape)
        # print(img_h)

        # camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

        # pred_vertices_2d = perspective_projection(
        #         pred_vertices_t,
        #         rotation=torch.eye(3, device=config.device).unsqueeze(0).expand(pred_vertices_t.shape[0], -1, -1),
        #         translation=cam_full,
        #         translation=cam_full,
        #         focal_length=focal_l,
        #         camera_center=camera_center)
        #
        # mean_2d, std_2d, var_2d = torch.mean(pred_vertices_2d), torch.std(pred_vertices_2d), torch.var(pred_vertices_2d)
        # pred_vertices_2d = (pred_vertices_2d - mean_2d)/std_2d
        # pred_vertices_2d = pred_vertices_2d.to(config.device)




        # print('pred_vertices_2d',pred_vertices_2d.shape)
        # print('pred_vertices_2d', pred_vertices_2d)
        # print('pred_vertices_t', pred_vertices_t.shape)
        # print('pred_vertices_t', pred_vertices_t)




        # pred_keypoints3d = pred_output.joints[:, :24, :]
        # print(pred_keypoints3d)
        # print('pred_keypoints3d', pred_keypoints3d)
        pred_key_points3d = pred_output.joints.cpu().numpy()
        print('pred_key_points3d', pred_key_points3d)

        rotation = torch.eye(3, device=config.device).unsqueeze(0).expand(pred_key_points3d.shape[0], -1, -1)
        translation = cam_full

        optimizer.zero_grad()

        texture_feature = model(img).to(config.device)
        _, _, img_h, img_w = img.shape
        texture = Parameter(texture_feature, requires_grad=True)

        # verts_uvs = aux.verts_uvs
        # faces_uvs = face.textures_idx
        # faces_uvs = faces_uvs.unsqueeze(0).to(config.device)
        tex_maps = texture.to(config.device)
        texture = torch.permute(tex_maps, (0, 2, 3, 1))
        print('texture', texture.shape)
        # torchtotensor(tex_maps)
        # tex = Textures(verts_uvs=pred_vertices_2d, faces_uvs=faces_uvs, maps=texture)
        print('texture', texture.shape)
        # uv = torch.randn(config.batch_size, pred_vertices_t.shape[1], 2)


        # v, f, _ = load_obj(obj)
        # faces = f.verts_idx
        #
        # obj_data = load_objs_as_meshes([obj], device=config.device)



        raster_settings = RasterizationSettings(
            image_size=(img_h, img_w),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # lights = PointLights(device=config.device, location=[[0.0, 0.0, -3.0]])
        lights = AmbientLights(ambient_color=((1, 1, 1),)).to(config.device)
        # focal_length = (362.03867, 362.03867)
        focal_length = (724.0773, 724.0773)
        principal_point = (256, 256)
        K = compute_k(focal_length, principal_point)
        K = torch.from_numpy(K).float().to(config.device)
        cameras = FoVPerspectiveCameras(fov=55, K=K[None]).to(config.device)
        # cameras = FoVPerspectiveCameras(device=config.device, R=rotation, T=translation)
        raster = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(config.device)
        renderer = MeshRenderer(rasterizer=raster, shader=SoftPhongShader(device=config.device, cameras=cameras, lights=lights))
        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=cameras,
        #         raster_settings=raster_settings
        #     ),
        #     shader=SoftPhongShader(
        #         device=config.device,
        #         cameras=cameras,
        #         lights=lights
        #     )
        # )

        # textures_sample = obj.textures
        # texture_feature = torch.permute(texture_feature,(0, 2, 3, 1))


        # print('pred_vertices_2d',pred_vertices_2d.shape)
        # faces = faces.unsqueeze(dim=0)
        # texture_map = TexturesUV(texture, faces.to(config.device), uv.to(config.device))
        # # TexturesVertex
        # print('faces',faces.shape)
        # print('verts', pred_vertices_t)
        # print('verts', pred_vertices_t.shape)
        # print(pred_vertices_t.shape)
        # print(texture.shape)
        #
        # print(texture.shape)
        print('verts', verts)
        print('verts', verts.shape)
        # ra_faces_uv = torch.reshape(ra_faces_uv, (batch, ra_faces_uv.shape[0], ra_faces_uv.shape[1]))
        # ra_verts_uv = torch.reshape(ra_verts_uv, (batch, ra_verts_uv.shape[0], ra_verts_uv.shape[1]))
        ra_faces_uv = ra_faces_uv.type_as(texture).long()
        # ra_verts_uv = ra_verts_uv.type_as(texture).long()
        # ra_verts_uv = ra_verts_uv.type_as(texture).long()
        textures = TexturesUV(maps=texture, faces_uvs=[ra_faces_uv for _ in range(batch)],
                              verts_uvs=[ra_verts_uv for _ in range(batch)]).to(config.device)
        # texture = TexturesUV(texture, faces_uvs=pred_vertices_t, verts_uvs=pred_vertices_t)
        # texture = Textures(verts_uvs=)
        # print(texture_feature.shape)

        # print(textures.shape)
        smpl_mesh = Meshes(
            verts=[pred_vertices_t[0,:,:].to(config.device)],
            faces=[faces.to(config.device)],
            textures=textures.to(config.device)
        )

        pytorch3d.io.save_obj('/media/mnt/Project/data/epoch%d_%d.obj'% (epoch + 1, idx), verts=pred_vertices_t[0,:,:], faces=faces, verts_uvs=ra_verts_uv, faces_uvs=ra_faces_uv, texture_map=texture.squeeze(0))
        fragments = raster(smpl_mesh)
        texels = smpl_mesh.sample_textures(fragments)
        out = texels[:, :, :, 0, :]
        out = out.permute(0, 3, 1, 2)

        images = renderer(smpl_mesh)
        img_ = images[0,:,:,:3].cpu().detach().numpy()
        print('img',img_)
        print('sdfjsdjfskdfjksdjfkefuiwoejfsldkjfksdjfsd')
        plt.imshow(img_)
        plt.show()
        print('images',images.shape)
        images = images.permute(0, 3, 1, 2)
        rgb_img = model2(images)
        torchtotensor(idx, rgb_img)
        torchtotensor2(idx, img)
        print('rgb_img', rgb_img.shape)
        loss_mse = criterion(rgb_img, img)
        loss_total = loss_mse.item()
        losses.append(loss_total)

        loss_mse.backward()
        optimizer.step()
        scheduler.step()

        print('[train] epoch:%d / loss:%f ' % (epoch + 1, loss_mse.item()))

        # save weights
        if (epoch + 1) % config.save_freq == 0:
            weights_file_name = "epoch%d.pth" % (epoch + 1)
            weights_file = os.path.join(config.snap_path, weights_file_name)
            img_file_name = "rgb_img_epoch%d.jpg" % (epoch + 1)
            img_save_file = os.path.join(config.save_img_path, img_file_name)
            save_image(rgb_img, img_save_file)
            tex_file_name = "tex_img_epoch%d.jpg" % (epoch + 1)
            tex_save_file = os.path.join(config.save_img_path, tex_file_name)
            save_image(out, tex_save_file)
            gt_file_name = "gt_img_epoch%d.jpg" % (epoch + 1)
            gt_save_file = os.path.join(config.save_img_path, gt_file_name)
            save_image(img, gt_save_file)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_mse
            }, weights_file)
            print('save weights of epoch %d' % (epoch + 1))

        return np.mean(losses)


def eval_epoch(config, epoch, model, model2, criterion, test_loader, smpl_model):
    with torch.no_grad():
        losses = []
        model.eval()

        with open(config.ra_body_path, 'rb') as f:
            this_dict = pickle.load(f)
        ra_faces_uv = this_dict['faces_uv'].to(config.device)
        ra_verts_uv = this_dict['verts_uv'].to(config.device)
        ra_faces = this_dict['faces'].to(config.device)

        for idx, data in tqdm(enumerate(test_loader)):
            img = data['img'].to(config.device)
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
            # pred_vertices_np = pred_vertices.cpu().numpy()
            faces_numpy = smpl_model.faces.astype(np.float)
            faces = torch.from_numpy(faces_numpy)
            faces = torch.tensor(faces, dtype=torch.int64).to(config.device)
            print('img', img.shape)

            verts, face, aux = load_obj(obj_file[0])
            verts = verts.to(config.device)
            verts = torch.tensor(verts, dtype=torch.float)

            # _, _, img_h, img_w = img.shape
            # print(img_h)
            #
            # img_h = torch.tensor([img.shape[2]]).to(config.device)
            # img_w = torch.tensor([img.shape[3]])
            # print(img_h.shape)
            # print(img_h)

            # camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

            # pred_vertices_2d = perspective_projection(
            #         pred_vertices_t,
            #         rotation=torch.eye(3, device=config.device).unsqueeze(0).expand(pred_vertices_t.shape[0], -1, -1),
            #         translation=cam_full,
            #         translation=cam_full,
            #         focal_length=focal_l,
            #         camera_center=camera_center)
            #
            # mean_2d, std_2d, var_2d = torch.mean(pred_vertices_2d), torch.std(pred_vertices_2d), torch.var(pred_vertices_2d)
            # pred_vertices_2d = (pred_vertices_2d - mean_2d)/std_2d
            # pred_vertices_2d = pred_vertices_2d.to(config.device)

            # print('pred_vertices_2d',pred_vertices_2d.shape)
            # print('pred_vertices_2d', pred_vertices_2d)
            # print('pred_vertices_t', pred_vertices_t.shape)
            # print('pred_vertices_t', pred_vertices_t)

            # pred_keypoints3d = pred_output.joints[:, :24, :]
            # print(pred_keypoints3d)
            # print('pred_keypoints3d', pred_keypoints3d)
            pred_key_points3d = pred_output.joints.cpu().numpy()
            print('pred_key_points3d', pred_key_points3d)

            rotation = torch.eye(3, device=config.device).unsqueeze(0).expand(pred_key_points3d.shape[0], -1, -1)
            translation = cam_full


            texture_feature = model(img).to(config.device)
            _, _, img_h, img_w = img.shape
            texture = Parameter(texture_feature, requires_grad=True)

            # verts_uvs = aux.verts_uvs
            # faces_uvs = face.textures_idx
            # faces_uvs = faces_uvs.unsqueeze(0).to(config.device)
            tex_maps = texture.to(config.device)
            texture = torch.permute(tex_maps, (0, 2, 3, 1))
            print('texture', texture.shape)
            # torchtotensor(tex_maps)
            # tex = Textures(verts_uvs=pred_vertices_2d, faces_uvs=faces_uvs, maps=texture)
            print('texture', texture.shape)
            # uv = torch.randn(config.batch_size, pred_vertices_t.shape[1], 2)

            # v, f, _ = load_obj(obj)
            # faces = f.verts_idx
            #
            # obj_data = load_objs_as_meshes([obj], device=config.device)

            raster_settings = RasterizationSettings(
                image_size=(img_h, img_w),
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # lights = PointLights(device=config.device, location=[[0.0, 0.0, -3.0]])
            lights = AmbientLights(ambient_color=((1, 1, 1),)).to(config.device)
            # focal_length = (362.03867, 362.03867)
            focal_length = (724.0773, 724.0773)
            principal_point = (256, 256)
            K = compute_k(focal_length, principal_point)
            K = torch.from_numpy(K).float().to(config.device)
            cameras = FoVPerspectiveCameras(fov=55, K=K[None]).to(config.device)
            # cameras = FoVPerspectiveCameras(device=config.device, R=rotation, T=translation)
            raster = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(config.device)
            renderer = MeshRenderer(rasterizer=raster,
                                    shader=SoftPhongShader(device=config.device, cameras=cameras, lights=lights))
            # renderer = MeshRenderer(
            #     rasterizer=MeshRasterizer(
            #         cameras=cameras,
            #         raster_settings=raster_settings
            #     ),
            #     shader=SoftPhongShader(
            #         device=config.device,
            #         cameras=cameras,
            #         lights=lights
            #     )
            # )

            # textures_sample = obj.textures
            # texture_feature = torch.permute(texture_feature,(0, 2, 3, 1))

            # print('pred_vertices_2d',pred_vertices_2d.shape)
            # faces = faces.unsqueeze(dim=0)
            # texture_map = TexturesUV(texture, faces.to(config.device), uv.to(config.device))
            # # TexturesVertex
            # print('faces',faces.shape)
            # print('verts', pred_vertices_t)
            # print('verts', pred_vertices_t.shape)
            # print(pred_vertices_t.shape)
            # print(texture.shape)
            #
            # print(texture.shape)
            print('verts', verts)
            print('verts', verts.shape)
            # ra_faces_uv = torch.reshape(ra_faces_uv, (batch, ra_faces_uv.shape[0], ra_faces_uv.shape[1]))
            # ra_verts_uv = torch.reshape(ra_verts_uv, (batch, ra_verts_uv.shape[0], ra_verts_uv.shape[1]))
            ra_faces_uv = ra_faces_uv.type_as(texture).long()
            # ra_verts_uv = ra_verts_uv.type_as(texture).long()
            # ra_verts_uv = ra_verts_uv.type_as(texture).long()
            textures = TexturesUV(maps=texture, faces_uvs=[ra_faces_uv for _ in range(batch)],
                                  verts_uvs=[ra_verts_uv for _ in range(batch)]).to(config.device)
            # texture = TexturesUV(texture, faces_uvs=pred_vertices_t, verts_uvs=pred_vertices_t)
            # texture = Textures(verts_uvs=)
            # print(texture_feature.shape)
            pred_ddd = pred_vertices_t[0, :, :]
            print(pred_ddd.shape)
            # print(textures.shape)
            smpl_mesh = Meshes(
                verts=[pred_vertices_t[0, :, :].to(config.device)],
                faces=[faces.to(config.device)],
                textures=textures.to(config.device)
            )

            # pytorch3d.io.save_obj('/media/mnt/Project/data/epoch%d_%d.obj' % (epoch + 1, idx), verts=pred_vertices_t[0, :, :], faces=faces,
            #                       verts_uvs=ra_verts_uv, faces_uvs=ra_faces_uv, texture_map=texture.squeeze(0))
            fragments = raster(smpl_mesh)
            texels = smpl_mesh.sample_textures(fragments)
            out = texels[:, :, :, 0, :]
            out = out.permute(0, 3, 1, 2)

            images = renderer(smpl_mesh)
            # img_ = images[0, :, :, :3].cpu().detach().numpy()
            # print('img', img_)
            # print('sdfjsdjfskdfjksdjfkefuiwoejfsldkjfksdjfsd')
            # plt.imshow(img_)
            # plt.show()
            print('images', images.shape)
            images = images.permute(0, 3, 1, 2)
            rgb_img = model2(images)
            # torchtotensor(idx, rgb_img)
            # torchtotensor2(idx, img)
            print('rgb_img', rgb_img.shape)
            loss_mse = criterion(rgb_img, img)
            loss_total = loss_mse.item()
            losses.append(loss_total)


        # for idx, data in tqdm(enumerate(test_loader)):
        #     img = data['img'].to(config.device)
        #     betas = data['betas'].to(config.device)
        #     rotmat = data['rotmat'].to(config.device)
        #     cam_full = data['cam_full'].to(config.device)
        #     joints = data['joints'].to(config.device)
        #     focal_l = data['focal_l'].to(config.device)
        #     detection_all = data['detection_all'].to(config.device)
        #     obj_file = data['obj_path']
        #
        #     pred_output = smpl_model(betas=betas,
        #                              body_pose=rotmat[:, 1:],
        #                              global_orient=rotmat[:, [0]],
        #                              pose2rot=True,
        #                              transl=cam_full)
        #     pred_vertices_t = pred_output.vertices
        #     # pred_vertices_np = pred_vertices.cpu().numpy()
        #     faces_numpy = smpl_model.faces.astype(np.float)
        #     faces = torch.from_numpy(faces_numpy)
        #     faces = torch.tensor(faces, dtype=torch.int64)
        #     print('img', img.shape)
        #
        #     verts, face, aux = load_obj(obj_file[0])
        #
        #     # _, _, img_h, img_w = img.shape
        #     # print(img_h)
        #
        #     img_h = torch.tensor([img.shape[2]])
        #     img_w = torch.tensor([img.shape[3]])
        #     print(img_h.shape)
        #     print(img_h)
        #
        #     camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2
        #
        #     # pred_vertices_2d = perspective_projection(
        #     #     pred_vertices_t,
        #     #     rotation=torch.eye(3, device=config.device).unsqueeze(0).expand(pred_vertices_t.shape[0], -1, -1),
        #     #     translation=cam_full,
        #     #     focal_length=focal_l,
        #     #     camera_center=camera_center)
        #
        #     mean_2d, std_2d, var_2d = torch.mean(pred_vertices_2d), torch.std(pred_vertices_2d), torch.var(
        #         pred_vertices_2d)
        #     pred_vertices_2d = (pred_vertices_2d - mean_2d) / std_2d
        #     pred_vertices_2d = pred_vertices_2d.to(config.device)
        #
        #
        #     pred_key_points3d = pred_output.joints.cpu().numpy()
        #     print('pred_key_points3d', pred_key_points3d)
        #
        #     rotation = torch.eye(3, device=config.device).unsqueeze(0).expand(pred_key_points3d.shape[0], -1, -1)
        #     translation = cam_full
        #
        #
        #
        #     texture_feature = model(img)
        #     _, _, img_h, img_w = img.shape
        #     texture = Parameter(texture_feature, requires_grad=True)
        #
        #     verts_uvs = aux.verts_uvs
        #     faces_uvs = faces.textures_idx
        #     faces_uvs = faces_uvs.unsqueeze(0).to(config.device)
        #     tex_maps = texture
        #     texture = torch.permute(tex_maps, (0, 2, 3, 1))
        #     print('texture', texture.shape)
        #
        #     tex = Textures(verts_uvs=pred_vertices_2d, faces_uvs=faces_uvs, maps=texture)
        #     print('texture', texture.shape)
        #
        #
        #     cameras = FoVPerspectiveCameras(device=config.device, R=rotation, T=translation)
        #     print(cameras)
        #     raster_settings = RasterizationSettings(
        #         image_size=(img_h, img_w),
        #         blur_radius=0.0,
        #         faces_per_pixel=1,
        #     )
        #
        #     lights = PointLights(device=config.device, location=[[0.0, 0.0, -3.0]])
        #
        #     renderer = MeshRenderer(
        #         rasterizer=MeshRasterizer(
        #             cameras=cameras,
        #             raster_settings=raster_settings
        #         ),
        #         shader=SoftPhongShader(
        #             device=config.device,
        #             cameras=cameras,
        #             lights=lights
        #         )
        #     )
        #     # textures_sample = obj.textures
        #     textures = Parameter(texture_feature, requires_grad=True)
        #     mesh = Meshes(
        #         verts=[verts.to(config.device)],
        #         faces=[faces.verts_idx.to(config.device)],
        #         textures=tex
        #     )
        #
        #     images = renderer(mesh)
        #     print('images', images.shape)
        #     images = images.permute(0, 3, 1, 2)
        #     rgb_img = model2(images)
        #     loss_mse = criterion(rgb_img, img)
        #     loss_total = loss_mse.item()
        #     losses.append(loss_total)

        print('[train] epoch:%d / loss:%f ' % (epoch + 1, loss_mse.item()))

        return np.mean(losses)








