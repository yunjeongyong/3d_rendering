import torch
import numpy as np
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

def rasterizer_setting(config, img_h, img_w):
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
    # cameras = FoVPerspectiveCameras(device=config.device, R=rotation, T=translation, K=K[None], fov=55)
    raster = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(config.device)
    renderer = MeshRenderer(rasterizer=raster, shader=SoftPhongShader(device=config.device, cameras=cameras, lights=lights))
    return raster
