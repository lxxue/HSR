import cv2
import imageio
import numpy as np
import skimage
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SfMPerspectiveCameras,
    SoftPhongShader,
)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as scipy_R
from torch.nn import functional as F


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb:  # [-1,1] --> [0,1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10.0 * torch.log10(mse)

    return psnr


def load_rgb(path, normalize_rgb=False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    if normalize_rgb:  # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.0
    img = img.transpose(2, 0, 1)
    return img


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        p = torch.eye(4, dtype=torch.float32, device=R.device).repeat(pose.shape[0], 1, 1)
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples), device=uv.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:, :4].detach())
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.to(x.device)
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z, device=z.device)), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3), cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r**2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print("BOUNDING SPHERE PROBLEM!")
        exit()

    sphere_intersections = (
        torch.sqrt(under_sqrt + 1e-8) * torch.tensor([-1.0, 1.0], device=under_sqrt.device)
        - ray_cam_dot
    )
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def get_smpl_intersection(cam_loc, ray_directions, smpl_mesh, interval_dist=0.1):
    # smpl mesh scaling or bounding box with scaling?
    bbox = smpl_mesh.apply_scale(1.2).bounding_box
    n_imgs, n_pix, _ = ray_directions.shape
    # smpl_mesh.apply_scale(1.1)

    device = ray_directions.device
    ray_dirs = ray_directions[0].clone().cpu().numpy()
    ray_origins = np.tile(cam_loc[0].clone().cpu().numpy(), n_pix).reshape(n_pix, 3)
    locations, index_ray, _ = bbox.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=False
    )
    mask_intersect = np.zeros(ray_dirs.shape[0], dtype=np.bool)

    mask_intersect[index_ray] = True
    unfinished_mask_start = torch.from_numpy(mask_intersect, device=device)
    intersect_dis = np.linalg.norm(ray_origins[index_ray] - locations, axis=1)

    curr_start_points = torch.zeros((n_pix, 3), device=device)
    curr_start_points[unfinished_mask_start] = torch.tensor(
        locations - interval_dist * ray_dirs[mask_intersect], device=device
    )
    acc_start_dis = torch.zeros(n_pix, device=device)
    acc_start_dis[unfinished_mask_start] = torch.tensor(
        intersect_dis - interval_dist, device=device
    )
    acc_end_dis = torch.zeros(n_pix, device=device)
    acc_end_dis[unfinished_mask_start] = torch.tensor(intersect_dis + interval_dist, device=device)

    min_dis = acc_start_dis.clone()
    max_dis = acc_end_dis.clone()
    return (
        curr_start_points,
        unfinished_mask_start,
        acc_start_dis,
        acc_end_dis,
        min_dis,
        max_dis,
    )


def get_bbox_intersection(cam_loc, ray_directions, smpl_mesh):
    # smpl mesh scaling or bounding box with scaling?
    bbox = smpl_mesh.apply_scale(1.5).bounding_box

    n_pix, _ = ray_directions.shape

    ray_dirs_np = ray_directions.clone().cpu().numpy()
    ray_origins_np = np.tile(cam_loc[0].clone().cpu().numpy(), n_pix).reshape(n_pix, 3)
    locations, index_ray, _ = bbox.ray.intersects_location(
        ray_origins=ray_origins_np, ray_directions=ray_dirs_np, multiple_hits=True
    )
    # strong assumption that either the ray has zero hit or two hits!
    num_ray_hits = locations.shape[0] // 2
    device = ray_directions.device
    if num_ray_hits == n_pix:
        dists = np.linalg.norm(ray_origins_np[index_ray] - locations, axis=1)
        # condition is that all rays intersect the bbox
        near = dists[:n_pix]
        far = dists[n_pix:]
        return torch.tensor(near, device=device), torch.tensor(far, device=device), None
    else:

        unhit_first_index_ray = set(np.arange(0, 512)).difference(index_ray[:num_ray_hits])
        unhit_second_index_ray = set(np.arange(0, 512)).difference(index_ray[num_ray_hits:])
        if unhit_first_index_ray != unhit_second_index_ray:
            import ipdb

            ipdb.set_trace()
        unhit_index_ray = list(unhit_first_index_ray)
        to_pad_index_ray = np.random.choice(index_ray[:num_ray_hits].shape[0], len(unhit_index_ray))

        near_hit_locations = np.zeros((n_pix, locations.shape[1]))
        near_hit_locations[index_ray[:num_ray_hits]] = locations[:num_ray_hits]
        # padding the invalid two hits
        near_hit_locations[unhit_index_ray] = locations[:num_ray_hits][to_pad_index_ray]

        far_hit_locations = np.zeros((n_pix, locations.shape[1]))
        far_hit_locations[index_ray[num_ray_hits:]] = locations[num_ray_hits:]
        # padding the invalid two hits
        far_hit_locations[unhit_index_ray] = locations[num_ray_hits:][to_pad_index_ray]

        near = np.linalg.norm(ray_origins_np - near_hit_locations, axis=1)

        far = np.linalg.norm(ray_origins_np - far_hit_locations, axis=1)

        ray_dirs_np[unhit_index_ray] = ray_dirs_np[to_pad_index_ray]

        padded_ray_dirs = torch.tensor(ray_dirs_np, device=device)
        return (
            torch.tensor(near, device=device),
            torch.tensor(far, device=device),
            padded_ray_dirs,
        )


def get_new_cam_pose_fvr(pose, rotation_angle_y):
    rot = scipy_R.from_euler("y", rotation_angle_y, degrees=True).as_matrix()  # start+i*(2)
    R, C = pose[:3, :3], pose[:3, 3]
    T = -R @ C
    temp_P = np.eye(4, dtype=np.float32)
    temp_P[:3, :3] = R
    temp_P[:3, 3] = T
    transform = np.eye(4)
    transform[:3, :3] = rot
    final_P = temp_P @ transform

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[:3, :3] = final_P[:3, :3]
    new_pose[:3, 3] = -np.linalg.inv(final_P[:3, :3]) @ final_P[:3, 3]
    return new_pose


# based on https://github.com/demul/extrinsic2pyramid/blob/main/util/camera_pose_visualizer.py
def create_camera_mesh(c2w, intrinsic=None, color=(1.0, 1.0, 1.0)):
    # cannot import open3d on euler
    import open3d as o3d

    if intrinsic is None:
        focal_len_scaled = 5
        aspect_ratio = 0.3
    else:
        focal_len_scaled = intrinsic[0, 0]
        aspect_ratio = intrinsic[0, 2] / intrinsic[1, 2]

    a = focal_len_scaled * aspect_ratio
    b = focal_len_scaled
    vertices_camera = np.array(
        [[0, 0, 0, 1], [a, -a, b, 1], [a, a, b, 1], [-a, a, b, 1], [-a, -a, b, 1]]
    )
    vertices_world = vertices_camera @ c2w.T
    vertices_world = vertices_world[:, :3]
    triangels = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world),
        triangles=o3d.utility.Vector3iVector(triangels),
    )
    return mesh


class Renderer:
    def __init__(self, focal_length=None, principal_point=None, img_size=None, cam_intrinsic=None):

        super().__init__()
        # img_size=[1080, 1920]
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        principal_point = [
            -(self.cam_intrinsic[0, 2] - self.image_size[1] / 2.0) / (self.image_size[1] / 2.0),
            -(self.cam_intrinsic[1, 2] - self.image_size[0] / 2.0) / (self.image_size[0] / 2.0),
        ]
        self.principal_point = torch.tensor(principal_point, device=self.device).unsqueeze(0)

        self.cam_R = (
            torch.from_numpy(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))
            .cuda()
            .float()
            .unsqueeze(0)
        )

        self.cam_T = torch.zeros((1, 3)).cuda().float()

        half_max_length = max(self.cam_intrinsic[0:2, 2])
        self.focal_length = torch.tensor(
            [
                (self.cam_intrinsic[0, 0] / half_max_length).astype(np.float32),
                (self.cam_intrinsic[1, 1] / half_max_length).astype(np.float32),
            ]
        ).unsqueeze(0)

        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            device=self.device,
        )

        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 0.0]],
            ambient_color=((1, 1, 1),),
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
        )
        # self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        self.raster_settings = RasterizationSettings(
            image_size=self.render_img_size,
            faces_per_pixel=10,
            blur_radius=0,
            max_faces_per_bin=30000,
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def set_camera(self, R, T):
        self.cam_R = R
        self.cam_T = T
        self.cam_R[:, :2, :] *= -1.0
        self.cam_T[:, :2] *= -1.0
        self.cam_R = torch.transpose(self.cam_R, 1, 2)
        self.cameras = SfMPerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            device=self.device,
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_recon(self, verts, faces, R=None, T=None, colors=None, mode="npat"):
        """
        mode: normal, phong, texture
        """
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = -torch.tensor([0, 0, -1]).float().to(verts.device)
            shades = (
                (normals * front_light.view(1, 1, 3))
                .sum(-1)
                .clamp(min=0)
                .unsqueeze(-1)
                .expand(-1, -1, 3)
            )
            results = []
            # shading
            if "p" in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)
            # normal
            if "n" in mode:
                # import pdb
                # pdb.set_trace()

                normals_vis = normals * 0.5 + 0.5  # -1*normals* 0.5 + 0.5
                normals_vis = normals_vis[:, :, [2, 1, 0]]
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # albedo
            if "a" in mode:
                assert colors is not None
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)

            # albedo*shading
            if "t" in mode:
                assert colors is not None
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return torch.cat(results, axis=1)


def render_trimesh(renderer, mesh, R, T, mode="np"):

    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, ..., :3] / 255
    renderer.set_camera(R, T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)

    return image
