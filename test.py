import torch
from torch.nn.parameter import Parameter
import os
import numpy as np
from models.smpl import SMPL
from common import constants
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
def main(args):
    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    file_dir = '/media/mnt/dataset/i_cliff_hr48.npz'
    obj_data_dir = '/media/mnt/dataset/obj_file'
    results = np.load(file_dir)
    smpl_model = SMPL(constants.SMPL_MODEL_DIR).to(device)

    pred_betas = torch.from_numpy(results['shape']).float().to(device)
    pred_rotmat = torch.from_numpy(results['pose']).float().to(device)
    pred_cam_full = torch.from_numpy(results['global_t']).float().to(device)
    pred_vert_arr = []
    pred_output = smpl_model(betas=pred_betas,
                             body_pose=pred_rotmat[:, 1:],
                             global_orient=pred_rotmat[:, [0]],
                             pose2rot=False,
                             transl=pred_cam_full)

    pred_vertices = pred_output.vertices
    pred_vert_arr.extend(pred_vertices.cpu().numpy())
    print(pred_vert_arr)

    pred_keypoints3d = pred_output.joints[:,:24,:]
    print('pred_keypoints3d',pred_keypoints3d)
    pred_key_points3d = pred_output.joints.cpu().numpy()
    print('pred_key_points3d', pred_key_points3d)

    rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
    translation = pred_cam_full

    verts_idx = []
    faces_idx = []

    obj_path = os.listdir(obj_data_dir)
    obj_whole_path = [os.path.join(obj_data_dir, obj_p) for obj_p in tqdm(obj_path)]
    obj_data_load = [load_objs_as_meshes(obj_pp, device=device) for obj_pp in tqdm(obj_whole_path)]
    for obj_pp in tqdm(obj_whole_path):
        v, f, _ = load_obj(obj_pp)
        verts_idx.append(v)
        faces_idx.append(f)
    # verts_faces_data_load = [load_obj(obj_pp, device=device) for obj_pp in tqdm(obj_whole_path)]


    cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translation)
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    texture_image_list = []
    for i, (obj, verts, faces) in enumerate(zip(obj_data_load, verts_idx, faces_idx)):
        textures_sample = obj.textures
        textures = Parameter(textures_sample, requires_grad=True)
        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )
        images = renderer(mesh)

    # Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print_hi('PyCharm')