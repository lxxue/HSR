import csv
import pickle
from pathlib import Path

import cv2
import kaolin
import numpy as np
import pytorch3d
import pytorch_lightning as pl
import torch
import torch.nn as nn
import trimesh
from einops import rearrange, repeat
from kaolin.ops.mesh import index_vertices_by_faces
from matplotlib import pyplot as plt
from torch.autograd import grad
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from hsr.model_components.body_model_params import BodyModelParams
from hsr.model_components.body_sampler import PointInSpace
from hsr.model_components.camera_pose_params import CameraPoseParams
from hsr.model_components.deformer import SMPLDeformer, skinning
from hsr.model_components.density import LaplaceDensity
from hsr.model_components.loss import S3IM, ScaleAndShiftInvariantLoss, compute_scale_and_shift
from hsr.model_components.network import ImplicitNet, RenderingNet
from hsr.model_components.ray_sampler import (
    ErrorBoundSamplerWithDeformerAndBBox,
    ErrorBoundSamplerWithoutDeformer,
)
from hsr.smpl import smpl_server
from hsr.utils import idr_utils, mesh_utils, render_utils


class HSRNetwork(nn.Module):
    def __init__(self, cfg, gender, num_training_frames, betas_path, exclude_frames):
        super().__init__()

        # Foreground object's networks
        self.fg_implicit_network = ImplicitNet(cfg.fg_implicit_network)
        self.fg_rendering_network = RenderingNet(cfg.fg_rendering_network)

        # Background object's networks
        self.bg_implicit_network = ImplicitNet(cfg.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(cfg.bg_rendering_network)

        self.bg_rendering_mode = cfg.bg_rendering_network.mode
        if "frame_encoding" in self.bg_rendering_mode:
            self.frame_encoder = nn.Embedding(
                num_training_frames, cfg.bg_rendering_network.dim_frame_encoding
            )
            self.dim_frame_encoding = cfg.bg_rendering_network.dim_frame_encoding
        self.exclude_frames = exclude_frames if exclude_frames else []

        betas = np.load(betas_path)
        scale = betas[0]
        betas = betas[1:]
        self.fg_deformer = SMPLDeformer(gender=gender, betas=betas, scale=scale)

        self.scene_bounding_sphere = cfg.fg_implicit_network.scene_bounding_sphere
        self.fg_density = LaplaceDensity(**cfg.fg_density)
        self.bg_density = LaplaceDensity(**cfg.bg_density)

        self.fg_ray_sampler = ErrorBoundSamplerWithDeformerAndBBox(
            self.scene_bounding_sphere, **cfg.fg_ray_sampler
        )
        self.bg_ray_sampler = ErrorBoundSamplerWithoutDeformer(
            self.scene_bounding_sphere, **cfg.bg_ray_sampler
        )
        self.eik_sampler = PointInSpace()
        self.smpl_server = smpl_server.SMPLServer(gender=gender, betas=betas, scale=scale)
        if cfg.smpl_init:
            smpl_model_state_path = Path(__file__).parents[1] / "checkpoints" / f"smpl_init.pth"
            smpl_model_state = torch.load(smpl_model_state_path)
            self.fg_implicit_network.load_state_dict(smpl_model_state["model_state_dict"])

        self.register_buffer("mesh_v_cano", self.smpl_server.verts_c)
        self.register_buffer(
            "mesh_f_cano", torch.tensor(self.smpl_server.smpl.faces.astype(np.int64))
        )
        self.register_buffer(
            "mesh_face_vertices",
            index_vertices_by_faces(self.mesh_v_cano, self.mesh_f_cano),
        )
        self.penetration_loss_steps = cfg.max_steps // 2

    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts):
        x_c, outlier_mask = self.fg_deformer.forward(
            x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts
        )
        output = self.fg_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        if not self.training:
            sdf[outlier_mask] = self.scene_bounding_sphere + 1.0

        return sdf, x_c, feature

    def sdf_func_without_smpl_deformer(self, x, cond):
        output = self.bg_implicit_network(x, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        return sdf, feature

    # determine if ray is in or off the surface, not points
    def check_off_in_surface_points_cano(self, x_cano, N_samples, threshold=0.05):
        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x_cano.unsqueeze(0).contiguous(), self.mesh_face_vertices
        )

        distance = torch.sqrt(distance + 1e-8)  # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(
            self.mesh_v_cano, self.mesh_f_cano, x_cano.unsqueeze(0)
        ).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.0).squeeze(1)
        return index_off_surface, index_in_surface

    def forward(self, inputs):
        torch.set_grad_enabled(True)
        intrinsics = inputs["intrinsics"]
        pose = inputs["pose"]
        uv = inputs["uv"]

        smpl_scale = inputs["smpl_scale"]
        smpl_pose = inputs["smpl_pose"]
        smpl_shape = inputs["smpl_shape"]
        smpl_trans = inputs["smpl_trans"]
        smpl_output = self.smpl_server(smpl_scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_output["smpl_tfs"]

        fg_implicit_cond = {"smpl": smpl_pose[:, 3:] / np.pi}
        # avoid shape overfitting to the pose
        if self.training:
            if inputs["current_epoch"] < 500 or inputs["current_epoch"] % 15 == 0:
                fg_implicit_cond = {"smpl": smpl_pose[:, 3:] * 0.0}
        # B x N x 3, B x 3
        ray_dirs, cam_loc = render_utils.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        ray_dirs_tmp, _ = render_utils.get_camera_params(
            uv, torch.eye(4).to(pose.device)[None], intrinsics
        )
        depth_scale = ray_dirs_tmp[0, :, 2:]

        ray_dirs = rearrange(ray_dirs, "b n d -> (b n) d")
        cam_loc = repeat(cam_loc, "b d -> (b n) d", n=num_pixels)

        # BN x M
        fg_z_vals = self.fg_ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self, fg_implicit_cond, smpl_tfs, smpl_output["smpl_verts"]
        )
        # BN x M, BN x 1
        bg_z_vals, bg_z_eik_vals = self.bg_ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        # BN x 3
        cam_loc = rearrange(cam_loc, "bn d -> bn 1 d")
        ray_dirs = rearrange(ray_dirs, "bn d -> bn 1 d")

        # BN x M x 3 = BN x 1 x 3 + BN x M x 1 * BN x 1 x 3
        fg_points = cam_loc + fg_z_vals[:, :, None] * ray_dirs
        num_samples = fg_points.shape[1]
        fg_points_flat = rearrange(fg_points, "bn m d -> (bn m) d")

        # BNM x 1, BNM x 3, BNM x D
        fg_sdf, fg_cano_points, fg_features = self.sdf_func_with_smpl_deformer(
            fg_points_flat, fg_implicit_cond, smpl_tfs, smpl_output["smpl_verts"]
        )

        # BN x M x 3
        bg_points = cam_loc + bg_z_vals[:, :, None] * ray_dirs
        assert bg_points.shape == fg_points.shape
        bg_points_flat = rearrange(bg_points, "bn m d -> (bn m) d")
        bg_implicit_cond = None
        # BNM x 1, BNM x D
        bg_sdf, bg_features = self.sdf_func_without_smpl_deformer(bg_points_flat, bg_implicit_cond)

        if self.training:
            if inputs["global_step"] > self.penetration_loss_steps:
                bg_samples_fg_sdf, _, _ = self.sdf_func_with_smpl_deformer(
                    bg_points_flat, fg_implicit_cond, smpl_tfs, smpl_output["smpl_verts"]
                )
                fg_samples_bg_sdf, _ = self.sdf_func_without_smpl_deformer(
                    fg_points_flat, bg_implicit_cond
                )
                bg_samples_fg_sdf = rearrange(
                    bg_samples_fg_sdf, "(b n m) 1 -> b n m", b=batch_size, n=num_pixels
                )
                fg_samples_bg_sdf = rearrange(
                    fg_samples_bg_sdf, "(b n m) 1 -> b n m", b=batch_size, n=num_pixels
                )
            else:
                bg_samples_fg_sdf = None
                fg_samples_bg_sdf = None

            index_off_surface, index_in_surface = self.check_off_in_surface_points_cano(
                fg_cano_points, num_samples
            )

            index_off_surface = rearrange(index_off_surface, "(b n) -> b n", b=batch_size)
            index_in_surface = rearrange(index_in_surface, "(b n) -> b n", b=batch_size)

            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server.verts_c.repeat(batch_size, 1, 1)
            indices = torch.randperm(smpl_verts_c.shape[1], device=smpl_verts_c.device)[:num_pixels]
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            # B x N x 3
            fg_eik_samples = self.eik_sampler.get_points(verts_c, global_ratio=0.0)
            fg_eik_samples.requires_grad_()
            fg_eik_sdf = self.fg_implicit_network(fg_eik_samples, fg_implicit_cond)[..., 0:1]
            fg_grad_theta = gradient(fg_eik_samples, fg_eik_sdf)

            bg_eik_uniform_samples = torch.empty(
                (batch_size, num_pixels, 3), device=smpl_verts_c.device
            )
            bg_eik_uniform_samples.uniform_(
                -self.scene_bounding_sphere / 2.0, self.scene_bounding_sphere / 2.0
            )
            bg_eik_near_samples = cam_loc + bg_z_eik_vals[:, :, None] * ray_dirs
            bg_eik_near_samples = rearrange(bg_eik_near_samples, "(b n) 1 d -> b n d", b=batch_size)
            bg_eik_near_samples = self.eik_sampler.get_points(bg_eik_near_samples, global_ratio=0.0)
            # bg_eik_near_samples = bg_eik_near_samples.reshape(-1, 3)
            # B x 2N x 3
            bg_eik_samples = torch.cat([bg_eik_uniform_samples, bg_eik_near_samples], dim=1)
            # bg_eik_samples = bg_eik_near_samples
            bg_eik_samples.requires_grad_()

            bg_eik_sdf = self.bg_implicit_network(bg_eik_samples, bg_implicit_cond)[..., 0:1]
            bg_grad_theta = gradient(bg_eik_samples, bg_eik_sdf)

        else:
            fg_grad_theta = None
            bg_grad_theta = None

        dirs = repeat(ray_dirs, "bn 1 d -> bn m d", m=num_samples)
        dirs = rearrange(dirs, "bn m d -> (bn m) d")
        view = -dirs

        # BNM x 3
        fg_rgb_flat, fg_others = self.get_fg_rgb_value(
            fg_points_flat,
            fg_cano_points,
            view,
            fg_implicit_cond,
            smpl_tfs,
            fg_features,
            self.training,
        )
        # BNM x 3
        fg_normals = fg_others["normals"]
        # B x D
        if "frame_encoding" in self.bg_rendering_mode:
            if inputs["idx"] not in self.exclude_frames:
                frame_code = self.frame_encoder(inputs["idx"])
            else:
                frame_code = torch.zeros((1, self.dim_frame_encoding), device=fg_normals.device)
        else:
            frame_code = None
        bg_rgb_flat, bg_others = self.get_bg_rgb_value(
            bg_points_flat, view, bg_features, frame_code, self.training
        )
        bg_normals = bg_others["normals"]

        fg_z_vals = rearrange(fg_z_vals, "bn m -> bn m 1")
        bg_z_vals = rearrange(bg_z_vals, "bn m -> bn m 1")
        fg_sdf = rearrange(fg_sdf, "(bn m) d -> bn m d", m=num_samples)
        bg_sdf = rearrange(bg_sdf, "(bn m) d -> bn m d", m=num_samples)
        fg_rgb = rearrange(fg_rgb_flat, "(bn m) d -> bn m d", m=num_samples)
        bg_rgb = rearrange(bg_rgb_flat, "(bn m) d -> bn m d", m=num_samples)
        fg_normals = rearrange(fg_normals, "(bn m) d -> bn m d", m=num_samples)
        bg_normals = rearrange(bg_normals, "(bn m) d -> bn m d", m=num_samples)

        rendering_outputs = self.fg_bg_volume_rendering(
            fg_z_vals, bg_z_vals, fg_sdf, bg_sdf, fg_rgb, bg_rgb, fg_normals, bg_normals
        )

        fg_sdf = rearrange(fg_sdf, "(b n) m 1 -> b n m", b=batch_size, n=num_pixels)
        bg_sdf = rearrange(bg_sdf, "(b n) m 1 -> b n m", b=batch_size, n=num_pixels)

        weights = rendering_outputs["weights"]
        # BN x M x 1 -> BN x 1
        acc_map = torch.sum(weights, 1)
        acc_map_fg = torch.sum(rendering_outputs["fg_weights"], 1)
        acc_map_bg = torch.sum(rendering_outputs["bg_weights"], 1)
        fg_acc_map = torch.sum(rendering_outputs["fg_weights_only"], 1)
        bg_acc_map = torch.sum(rendering_outputs["bg_weights_only"], 1)
        rgb = torch.sum(weights * rendering_outputs["rgb"], 1)
        fg_rgb = torch.sum(rendering_outputs["fg_weights_only"] * fg_rgb, 1)
        bg_rgb = torch.sum(rendering_outputs["bg_weights_only"] * bg_rgb, 1)
        normal = torch.sum(weights * rendering_outputs["normals"], 1)
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal = rot @ normal.permute(1, 0)
        normal = normal.permute(1, 0).contiguous()
        # renormalize the normals since volume integration might break the normalization
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        depth = torch.sum(weights * rendering_outputs["z_vals"], 1)
        # we should scale rendered distance to depth along z direction
        depth = depth_scale * depth

        rgb = rearrange(rgb, "(b n) c -> b n c", b=batch_size)
        fg_rgb = rearrange(fg_rgb, "(b n) c -> b n c", b=batch_size)
        bg_rgb = rearrange(bg_rgb, "(b n) c -> b n c", b=batch_size)
        normal = rearrange(normal, "(b n) c -> b n c", b=batch_size)
        depth = rearrange(depth, "(b n) 1 -> b n", b=batch_size)
        acc_map = rearrange(acc_map, "(b n) 1 -> b n", b=batch_size)
        acc_map_fg = rearrange(acc_map_fg, "(b n) 1 -> b n", b=batch_size)
        acc_map_bg = rearrange(acc_map_bg, "(b n) 1 -> b n", b=batch_size)
        fg_acc_map = rearrange(fg_acc_map, "(b n) 1 -> b n", b=batch_size)
        bg_acc_map = rearrange(bg_acc_map, "(b n) 1 -> b n", b=batch_size)

        if self.training:
            output = {
                "rgb": rgb,
                "fg_rgb": fg_rgb,
                "bg_rgb": bg_rgb,
                "normal": normal,
                "depth": depth,
                "index_outside": inputs["index_outside"],
                "index_off_surface": index_off_surface,
                "index_in_surface": index_in_surface,
                "acc_map": acc_map,
                "fg_acc_map": fg_acc_map,
                "bg_acc_map": bg_acc_map,
                "acc_map_fg": acc_map_fg,
                "acc_map_bg": acc_map_bg,
                "fg_sdf": fg_sdf,
                "bg_sdf": bg_sdf,
                "bg_samples_fg_sdf": bg_samples_fg_sdf,
                "fg_samples_bg_sdf": fg_samples_bg_sdf,
                "fg_grad_theta": fg_grad_theta,
                "bg_grad_theta": bg_grad_theta,
            }
        else:
            fg_normal = torch.sum(rendering_outputs["fg_weights_only"] * fg_normals, 1)
            bg_normal = torch.sum(rendering_outputs["bg_weights_only"] * bg_normals, 1)
            fg_normal = rot @ fg_normal.permute(1, 0)
            fg_normal = fg_normal.permute(1, 0).contiguous()
            bg_normal = rot @ bg_normal.permute(1, 0)
            bg_normal = bg_normal.permute(1, 0).contiguous()
            fg_depth = torch.sum(rendering_outputs["fg_weights_only"] * fg_z_vals, 1)
            fg_depth = depth_scale * fg_depth
            bg_depth = torch.sum(rendering_outputs["bg_weights_only"] * bg_z_vals, 1)
            bg_depth = depth_scale * bg_depth
            fg_depth = rearrange(fg_depth, "(b n) 1 -> b n", b=batch_size)
            bg_depth = rearrange(bg_depth, "(b n) 1 -> b n", b=batch_size)
            fg_normal = rearrange(fg_normal, "(b n) m -> b n m", b=batch_size)
            bg_normal = rearrange(bg_normal, "(b n) m -> b n m", b=batch_size)
            # calculate entropy for uncertainty measure
            fg_weights = rendering_outputs["fg_weights"]
            fg_weights = rearrange(fg_weights, "bn m 1 -> bn m")
            entropy = torch.distributions.Categorical(probs=fg_weights + 1e-12).entropy()
            negative_entropy = -entropy
            # maximum entropy: -1/b * np.log(1/b) * b with b = 162 (128 + 32 + 2)
            negative_entropy /= 5.087
            # map from [-1, 0] to [0, 1]
            negative_entropy += 1.0
            negative_entropy[torch.sum(fg_weights, 1) < 0.9] = 0.0
            negative_entropy = rearrange(negative_entropy, "(b n) -> b n", b=batch_size)
            output = {
                "fg_acc_map": fg_acc_map,
                "bg_acc_map": bg_acc_map,
                "rgb": rgb,
                "fg_rgb": fg_rgb,
                "bg_rgb": bg_rgb,
                "normal": normal,
                "fg_normal": fg_normal,
                "bg_normal": bg_normal,
                "depth": depth,
                "fg_depth": fg_depth,
                "bg_depth": bg_depth,
                "negative_entropy": negative_entropy,
            }
        return output

    def get_fg_rgb_value(
        self,
        x,
        points,
        view_dirs,
        cond,
        tfs,
        feature_vectors,
        surface_body_parsing=None,
        is_training=True,
    ):
        pnts_c = points
        others = {}

        _, gradients, feature_vectors = self.fg_forward_gradient(
            x, pnts_c, cond, tfs, create_graph=is_training, retain_graph=is_training
        )
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        fg_rendering_output = self.fg_rendering_network(
            pnts_c,
            normals,
            view_dirs,
            cond["smpl"],
            feature_vectors,
            surface_body_parsing,
        )

        rgb_vals = fg_rendering_output[:, :3]
        others["normals"] = normals
        return rgb_vals, others

    def get_bg_rgb_value(
        self,
        x,
        view_dirs,
        bg_feature_vectors,
        frame_code=None,
        is_training=True,
    ):
        others = {}

        bg_gradients, bg_feature_vectors = self.bg_forward_gradient(
            x, create_graph=is_training, retain_graph=is_training
        )
        bg_normals = nn.functional.normalize(bg_gradients, dim=-1, eps=1e-6)
        bg_rendering_output = self.bg_rendering_network(
            x,
            bg_normals,
            view_dirs,
            None,
            bg_feature_vectors,
            frame_latent_code=frame_code,
            surface_body_parsing=None,
        )

        bg_rgb_vals = bg_rendering_output[:, :3]
        others["normals"] = bg_normals
        return bg_rgb_vals, others

    def fg_forward_gradient(self, x, pnts_c, cond, tfs, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        pnts_d = self.fg_deformer.forward_skinning(pnts_c[None], None, tfs)[0]
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True,
            )[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.fg_implicit_network(pnts_c, cond)[0]
        sdf = output[:, :1]

        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]

        return (
            grads.reshape(grads.shape[0], -1),
            torch.einsum("bi,bij->bj", gradients, grads_inv),
            feature,
        )

    def bg_forward_gradient(self, x, create_graph=True, retain_graph=True):
        x.requires_grad_(True)
        output = self.bg_implicit_network(x, None)[0]
        sdf = output[:, :1]
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        return gradients, feature

    def get_weights(self, density, dists):
        # LOG SPACE
        free_energy = dists * density[:, :, 0]
        # add 0 for transperancy 1 at t_0
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                free_energy[:, :-1],
            ],
            dim=-1,
        )
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        # probability of everything is empty up to now
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here
        return weights[:, :, None]

    def fg_bg_volume_rendering(
        self, fg_z_vals, bg_z_vals, fg_sdf, bg_sdf, fg_rgb, bg_rgb, fg_normals, bg_normals
    ):
        # z_vals: BN x M x 1, sdf: BN x M x 1, rgb: BN x M x 3, normals: BN x M x 3
        # density: BN x M x 1
        fg_density = self.fg_density(fg_sdf)
        bg_density = self.bg_density(bg_sdf)

        # BN x M x D
        z_vals = torch.cat([fg_z_vals, bg_z_vals], dim=1)
        density = torch.cat([fg_density, bg_density], dim=1)
        rgb = torch.cat([fg_rgb, bg_rgb], dim=1)
        normals = torch.cat([fg_normals, bg_normals], dim=1)
        z_vals, sort_idx = torch.sort(z_vals, dim=1)
        density = torch.gather(density, 1, sort_idx)
        sort_idx = repeat(sort_idx, "bn m 1 -> bn m d", d=3)
        rgb = torch.gather(rgb, 1, sort_idx)
        normals = torch.gather(normals, 1, sort_idx)

        dists = z_vals[:, 1:, 0] - z_vals[:, :-1, 0]
        last_dist = torch.full((z_vals.shape[0], 1), 1e10, device=z_vals.device)
        dists = torch.cat([dists, last_dist], -1)
        fg_dists = fg_z_vals[:, 1:, 0] - fg_z_vals[:, :-1, 0]
        # do not concate infty here
        fg_dists = torch.cat([fg_dists, fg_dists[:, -1:]], -1)
        bg_dists = bg_z_vals[:, 1:, 0] - bg_z_vals[:, :-1, 0]
        bg_dists = torch.cat([bg_dists, last_dist], -1)
        weights = self.get_weights(density, dists)
        fg_weights_only = self.get_weights(fg_density, fg_dists)
        bg_weights_only = self.get_weights(bg_density, bg_dists)

        # BNxMxD -> BNxMx1
        sort_idx = sort_idx[:, :, 0:1]
        inverse_sort_idx = torch.argsort(sort_idx, dim=1)
        num_samples = z_vals.shape[1] // 2
        assert z_vals.shape[1] == num_samples * 2
        fg_weights = torch.gather(weights, 1, inverse_sort_idx[:, :num_samples, :])
        bg_weights = torch.gather(weights, 1, inverse_sort_idx[:, num_samples:, :])

        outputs = {
            "z_vals": z_vals,
            "rgb": rgb,
            "normals": normals,
            "weights": weights,
            "fg_weights": fg_weights,
            "bg_weights": bg_weights,
            "fg_weights_only": fg_weights_only,
            "bg_weights_only": bg_weights_only,
        }
        return outputs


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, :, -3:]
    return points_grad


class HSR(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.training_indices = list(
            range(
                cfg.dataset.train.start_frame,
                cfg.dataset.train.end_frame + 1,
                cfg.dataset.train.skip_step,
            )
        )
        exclude_frames = cfg.dataset.train.exclude_frames
        if exclude_frames is not None:
            for i in exclude_frames:
                self.training_indices.remove(i)
        num_training_frames = len(self.training_indices)

        data_dir = cfg.dataset.train.data_dir
        betas_path = data_dir / "mean_shape.npy"
        self.model = HSRNetwork(
            cfg.model, cfg.dataset.gender, num_training_frames, betas_path, exclude_frames
        )
        self.cfg = cfg
        self.opt_smpl = cfg.model.opt_smpl
        self.opt_camera_pose = cfg.model.opt_camera_pose
        self.val_render_image = cfg.model.val_render_image
        self.test_render_image = cfg.model.test_render_image
        self.test_save_numerical_results = cfg.model.test_save_numerical_results
        self.pixel_per_batch = cfg.dataset.val.pixel_per_batch
        self.update_canonical_mesh = cfg.model.update_canonical_mesh
        if self.opt_smpl:
            self.body_model_params = BodyModelParams(num_training_frames, model_type="smpl")
            self.load_body_model_params()
            optim_params = self.body_model_params.param_names
            for param_name in optim_params:
                self.body_model_params.set_requires_grad(param_name, requires_grad=True)

        if self.opt_camera_pose:
            self.camera_pose_params = CameraPoseParams(num_training_frames)
            self.load_camera_pose_params()

        self.loss = HSRLoss(cfg.model.loss)
        pcd_path = (
            data_dir / "normalized" / "colmap_scene_mesh_resampled_pcd_filtered_normalized.ply"
        )

        half_length = 0.8 * cfg.model.bg_implicit_network.scene_bounding_sphere
        scene_mesh_verts = torch.tensor(
            [
                [-half_length, -half_length, -half_length],
                [half_length, half_length, half_length],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("scene_mesh_verts", scene_mesh_verts)
        Path("rgb").mkdir(parents=False, exist_ok=True)
        Path("normal").mkdir(parents=False, exist_ok=True)
        Path("depth").mkdir(parents=False, exist_ok=True)
        Path("meshes").mkdir(parents=False, exist_ok=True)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        # colormaps
        self.sequential_colormap = plt.cm.get_cmap("inferno")
        self.diverging_colormap = plt.cm.get_cmap("seismic")
        self.depth_colormap = plt.cm.get_cmap("viridis")

    def load_camera_pose_params(self):
        data_root = self.cfg.dataset.train.data_dir
        c2ws = np.load(data_root / "c2ws.npy")
        c2ws = c2ws[self.training_indices]
        c2ws = torch.from_numpy(c2ws).float()

        # https://github.com/facebookresearch/pytorch3d/issues/1488
        c2ws = c2ws.transpose(-2, -1)
        log_reps = pytorch3d.transforms.se3_log_map(c2ws)
        self.camera_pose_params.se3_log_reps.weight.data = log_reps
        self.camera_pose_params.se3_log_reps.weight.requires_grad = True

    def load_body_model_params(self):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        data_root = self.cfg.dataset.train.data_dir
        mean_shape = np.load(data_root / "mean_shape.npy")
        scale_mat_scale = np.load(data_root / "cameras_normalize.npz")["scale_mat_0"][0, 0]
        body_model_params["scale"] = torch.tensor(
            mean_shape[0].reshape(1, 1) / scale_mat_scale,
            dtype=torch.float32,
        )

        body_model_params["betas"] = torch.tensor(
            mean_shape[1:][None],
            dtype=torch.float32,
        )
        body_model_params["global_orient"] = torch.tensor(
            np.load(data_root / "poses.npy")[self.training_indices][:, :3],
            dtype=torch.float32,
        )
        body_model_params["body_pose"] = torch.tensor(
            np.load(data_root / "poses.npy")[self.training_indices][:, 3:],
            dtype=torch.float32,
        )
        body_model_params["transl"] = torch.tensor(
            np.load(data_root / "normalize_trans.npy")[self.training_indices],
            dtype=torch.float32,
        )

        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(
                param_name, body_model_params[param_name], requires_grad=False
            )

    def configure_optimizers(self):
        params = []
        for name, module in self.model.named_children():
            if name == "fg_implicit_network":
                # reduce learning rate for the human density network to avoid large shape changes
                params.append({"params": module.parameters(), "lr": self.cfg.model.learning_rate})
            elif name == "body_model_params" or name == "camera_pose_params":
                assert False
            else:
                params.append({"params": module.parameters(), "lr": self.cfg.model.learning_rate})
        if self.opt_smpl:
            params.append(
                {
                    "params": self.body_model_params.parameters(),
                    "lr": self.cfg.model.learning_rate * 0.1,
                }
            )
        if self.opt_camera_pose:
            params.append(
                {
                    "params": self.camera_pose_params.parameters(),
                    "lr": self.cfg.model.learning_rate,
                }
            )

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.cfg.model.decay_rate ** (1 / self.cfg.model.decay_steps)
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch):
        inputs, targets = batch
        batch_idx = inputs["idx"]

        if self.opt_smpl:
            body_model_params = self.body_model_params(batch_idx)
            inputs["smpl_pose"] = torch.cat(
                (body_model_params["global_orient"], body_model_params["body_pose"]),
                dim=1,
            )
            inputs["smpl_shape"] = body_model_params["betas"]
            inputs["smpl_trans"] = body_model_params["transl"]
            inputs["smpl_scale"] = body_model_params["scale"]
            self.log("smpl_scale", body_model_params["scale"], prog_bar=True, on_step=True)
        else:
            inputs["smpl_pose"] = inputs["smpl_params"][:, 4:76]
            inputs["smpl_shape"] = inputs["smpl_params"][:, 76:]
            inputs["smpl_trans"] = inputs["smpl_params"][:, 1:4]
            inputs["smpl_scale"] = inputs["smpl_params"][:, 0:1]

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(batch_idx)

        inputs["current_epoch"] = self.current_epoch
        inputs["global_step"] = self.global_step
        model_outputs = self.model(inputs)
        model_outputs["epoch"] = self.current_epoch

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            self.log(k, v.item(), prog_bar=True, on_step=True)

        return loss_output["train/loss"]

    def query_fg_oc(self, x, cond):
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.fg_implicit_network(x, cond)[:, :, 0].reshape(-1, 1)
        return {"occ": mnfld_pred}

    def query_bg_oc(self, x, cond):
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.bg_implicit_network(x, cond)[:, :, 0].reshape(-1, 1)
        return {"occ": mnfld_pred}

    def query_bg_sdf(self, x, cond):
        x = x.reshape(-1, 3)
        sdf = self.model.bg_implicit_network(x, cond)[:, :, 0].reshape(-1)
        return sdf

    def query_wc(self, x, cond):
        x = x.reshape(-1, 3)
        w = self.model.fg_deformer.query_weights(x, cond)
        return w

    def query_fg_od(self, x, cond, smpl_tfs, smpl_verts):
        x = x.reshape(-1, 3)
        x_c, _ = self.model.fg_deformer.forward(
            x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts
        )
        output = self.model.fg_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        return {"occ": sdf}

    def query_od(self, x, cond, smpl_tfs, smpl_verts):
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformer.forward(
            x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts
        )
        output = self.model.implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        return {"occ": sdf}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def validation_step(self, batch, *args, **kwargs):
        outputs = {}
        inputs, targets = batch
        batch_idx = inputs["idx"]

        if self.opt_smpl:
            body_model_params = self.body_model_params(batch_idx)
            inputs["smpl_pose"] = torch.cat(
                (body_model_params["global_orient"], body_model_params["body_pose"]),
                dim=1,
            )
            inputs["smpl_shape"] = body_model_params["betas"]
            inputs["smpl_trans"] = body_model_params["transl"]
            inputs["smpl_scale"] = body_model_params["scale"]
        else:
            inputs["smpl_pose"] = inputs["smpl_params"][:, 4:76]
            inputs["smpl_shape"] = inputs["smpl_params"][:, 76:]
            inputs["smpl_trans"] = inputs["smpl_params"][:, 1:4]
            inputs["smpl_scale"] = inputs["smpl_params"][:, 0:1]

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(batch_idx)

        cond = {"smpl": inputs["smpl_pose"][:, 3:] / np.pi}
        mesh_canonical = mesh_utils.generate_mesh(
            lambda x: self.query_fg_oc(x, cond),
            self.model.smpl_server.verts_c[0],
            point_batch=10000,
            res_up=3,
        )
        verts_mesh = torch.tensor(
            mesh_canonical.vertices, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mesh_canonical = trimesh.Trimesh(
            mesh_canonical.vertices,
            mesh_canonical.faces,
        )

        smpl_outputs = self.model.smpl_server(
            inputs["smpl_scale"], inputs["smpl_trans"], inputs["smpl_pose"], inputs["smpl_shape"]
        )
        smpl_tfs = smpl_outputs["smpl_tfs"]
        verts_deformed = self.get_deformed_mesh(mesh_canonical.vertices, cond, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces)
        mesh_scene = mesh_utils.get_surface_sliding(
            lambda x: self.query_bg_sdf(x, None), resolution=512, grid_boundary=[-3.0, 3.0]
        )

        outputs["mesh_canonical"] = mesh_canonical
        outputs["mesh_deformed"] = mesh_deformed
        outputs["mesh_scene"] = mesh_scene

        if not self.val_render_image:
            return outputs

        total_pixels = inputs["total_pixels"][0]
        split = idr_utils.split_input(
            inputs,
            total_pixels,
            n_pixels=min(
                self.pixel_per_batch,
                targets["img_size"][0] * targets["img_size"][1],
            ),
        )

        res = []
        for s in split:
            out = self.model(s)
            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v
            res.append(
                {
                    "rgb": out["rgb"].detach(),
                    "fg_rgb": out["fg_rgb"].detach(),
                    "bg_rgb": out["bg_rgb"].detach(),
                    "negative_entropy": out["negative_entropy"].detach(),
                    "fg_acc_map": out["fg_acc_map"].detach(),
                    "normal": out["normal"].detach(),
                    "fg_normal": out["fg_normal"].detach(),
                    "bg_normal": out["bg_normal"].detach(),
                    "depth": out["depth"].detach(),
                    "fg_depth": out["fg_depth"].detach(),
                    "bg_depth": out["bg_depth"].detach(),
                }
            )
        batch_size = targets["rgb"].shape[0]

        model_outputs = idr_utils.merge_output(res, total_pixels, batch_size)

        outputs.update(
            {
                "idx": inputs["idx"],
                "rgb_pred": model_outputs["rgb"].detach(),
                "fg_rgb_pred": model_outputs["fg_rgb"].detach(),
                "bg_rgb_pred": model_outputs["bg_rgb"].detach(),
                "normal_pred": model_outputs["normal"].detach(),
                "fg_normal_pred": model_outputs["fg_normal"].detach(),
                "bg_normal_pred": model_outputs["bg_normal"].detach(),
                "depth_pred": model_outputs["depth"].detach().clone(),
                "fg_depth_pred": model_outputs["fg_depth"].detach(),
                "bg_depth_pred": model_outputs["bg_depth"].detach(),
                "fg_acc_map": model_outputs["fg_acc_map"].detach(),
                "negative_entropy": model_outputs["negative_entropy"].detach(),
                **targets,
            }
        )

        self.val_output_list.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        output = self.val_output_list[0]
        idx = output["idx"].item()
        mesh_canonical = output["mesh_canonical"]
        mesh_canonical.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_human.ply")
        mesh_deformed = output["mesh_deformed"]
        mesh_deformed.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_human_posed.ply")
        mesh_scene = output["mesh_scene"]
        mesh_scene.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_scene.ply")

        if self.update_canonical_mesh:
            self.model.mesh_v_cano = torch.tensor(
                mesh_canonical.vertices[None], device=self.device
            ).float()
            self.model.mesh_f_cano = torch.tensor(
                mesh_canonical.faces.astype(np.int64), device=self.device
            )
            self.model.mesh_face_vertices = index_vertices_by_faces(
                self.model.mesh_v_cano, self.model.mesh_f_cano
            )

        if not self.val_render_image:
            return
        img_size = output["img_size"]
        h, w = img_size
        concat_dim = 1 if h > w else 0

        depth_pred = output["depth_pred"]
        depth_pred = depth_pred.reshape(*img_size, -1)
        fg_depth_pred = output["fg_depth_pred"]
        fg_depth_pred = fg_depth_pred.reshape(*img_size, -1)
        bg_depth_pred = output["bg_depth_pred"]
        bg_depth_pred = bg_depth_pred.reshape(*img_size, -1)

        rgb_pred = output["rgb_pred"]
        rgb_pred = rgb_pred.reshape(*img_size, -1)
        rgb_gt = output["rgb"]
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        rgb = torch.cat([rgb_gt, rgb_pred], dim=concat_dim).cpu().numpy()
        rgb_error = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        rgb_error *= 5
        rgb_error = self.sequential_colormap(rgb_error.cpu().numpy())[:, :, :3]
        # rgb = np.concatenate([rgb, rgb_error], axis=concat_dim)
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb_pred = output["fg_rgb_pred"]
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
        fg_rgb = torch.cat([rgb_gt, fg_rgb_pred], dim=concat_dim).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        bg_rgb_pred = output["bg_rgb_pred"]
        bg_rgb_pred = bg_rgb_pred.reshape(*img_size, -1)
        bg_rgb = torch.cat([rgb_gt, bg_rgb_pred], dim=concat_dim).cpu().numpy()
        bg_rgb = (bg_rgb * 255).astype(np.uint8)

        fg_acc_map = output["fg_acc_map"]
        fg_acc_map = fg_acc_map.reshape(*img_size, -1)
        fg_acc_map = fg_acc_map.repeat(1, 1, 3)
        fg_acc_map = torch.cat([rgb_gt * fg_acc_map, fg_acc_map], dim=concat_dim).cpu().numpy()
        fg_acc_map = (fg_acc_map * 255).astype(np.uint8)

        negative_entropy = output["negative_entropy"]
        negative_entropy = negative_entropy.reshape(*img_size, -1)
        negative_entropy = negative_entropy.repeat(1, 1, 3)
        negative_entropy = (
            torch.cat([rgb_gt * negative_entropy, negative_entropy], dim=concat_dim).cpu().numpy()
        )
        negative_entropy = (negative_entropy * 255).astype(np.uint8)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_pred = torch.moveaxis(rgb_pred, -1, 0)[None, ...]
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        psnr = self.psnr(rgb_pred, rgb_gt)
        ssim = self.ssim(rgb_pred, rgb_gt)
        lpips = self.lpips(rgb_pred, rgb_gt)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True, prog_bar=True)

        normal_pred = output["normal_pred"].reshape(*img_size, -1)
        fg_normal_pred = output["fg_normal_pred"].reshape(*img_size, -1)
        bg_normal_pred = output["bg_normal_pred"].reshape(*img_size, -1)
        normal_gt = output["normal"].reshape(*img_size, -1)
        normal_error = 1.0 - torch.sum(normal_pred * normal_gt, dim=-1)

        normal = torch.cat([normal_gt, normal_pred], dim=concat_dim).cpu().numpy()
        normal = (normal + 1.0) * 0.5
        normal_cos = normal_error.mean()
        # map to [0, 1]
        normal_error *= 0.5
        # for visualization
        normal_error *= 5
        normal_error = self.sequential_colormap(normal_error.cpu().numpy())[:, :, :3]
        # normal = np.concatenate([normal, normal_error], axis=concat_dim)
        normal = (normal * 255).astype(np.uint8)
        fg_normal = torch.cat([normal_gt, fg_normal_pred], dim=concat_dim).cpu().numpy()
        fg_normal = (fg_normal + 1.0) * 0.5
        fg_normal = (fg_normal * 255).astype(np.uint8)
        bg_normal = torch.cat([normal_gt, bg_normal_pred], dim=concat_dim).cpu().numpy()
        bg_normal = (bg_normal + 1.0) * 0.5
        bg_normal = (bg_normal * 255).astype(np.uint8)

        self.log("val/normal_cos", normal_cos, on_step=False, on_epoch=True, prog_bar=True)

        depth_gt = output["depth"]
        depth_gt = depth_gt.reshape(*img_size, -1)

        depth_gt = rearrange(depth_gt, "h w 1 -> 1 h w")
        depth_pred = rearrange(depth_pred, "h w 1 -> 1 h w")
        mask = torch.ones_like(depth_gt)
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_l1 = torch.abs(depth_gt - depth_pred * scale - shift).mean()
        depth_pred_aligned = depth_pred * scale + shift
        self.log("val/depth_l1", depth_l1, on_step=False, on_epoch=True, prog_bar=True)

        depth_gt = rearrange(depth_gt, "1 h w -> h w")
        depth_pred = rearrange(depth_pred, "1 h w -> h w")
        depth_pred_aligned = rearrange(depth_pred_aligned, "1 h w -> h w")
        depth = torch.cat([depth_gt, depth_pred_aligned], dim=concat_dim).cpu().numpy()
        vis_scale = 1.0 / (depth.max() - depth.min())
        vis_shift = -depth.min()
        depth = (depth + vis_shift) * vis_scale
        depth = self.depth_colormap(depth)[:, :, :3]
        depth_error = depth_pred - depth_gt
        depth_error = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
        depth_error = self.diverging_colormap(depth_error.cpu().numpy())[:, :, :3]
        # depth = np.concatenate([depth, depth_error], axis=concat_dim)
        depth = (depth * 255).astype(np.uint8)
        fg_depth_aligned = fg_depth_pred * scale + shift
        fg_depth_aligned = rearrange(fg_depth_aligned, "h w 1 -> h w")
        fg_depth = torch.cat([depth_gt, fg_depth_aligned], dim=concat_dim).cpu().numpy()
        fg_depth = (fg_depth + vis_shift) * vis_scale
        fg_depth = self.depth_colormap(fg_depth)[:, :, :3]
        fg_depth = (fg_depth * 255).astype(np.uint8)
        bg_depth_aligned = bg_depth_pred * scale + shift
        bg_depth_aligned = rearrange(bg_depth_aligned, "h w 1 -> h w")
        bg_depth = torch.cat([depth_gt, bg_depth_aligned], dim=concat_dim).cpu().numpy()
        bg_depth = (bg_depth + vis_shift) * vis_scale
        bg_depth = self.depth_colormap(bg_depth)[:, :, :3]
        bg_depth = (bg_depth * 255).astype(np.uint8)

        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_mask.png", fg_acc_map[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_normal[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_normal[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}.png", depth[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_depth[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_depth[:, :, ::-1])

        self.val_output_list.clear()
        return

    def on_test_start(self):
        prefix = f"epoch_{self.current_epoch:04d}"
        Path(prefix).mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_mask").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_entropy").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_canonical").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_deformed").mkdir(parents=False, exist_ok=True)
        scene_mesh = mesh_utils.generate_mesh(
            lambda x: self.query_bg_oc(x, None),
            verts=self.scene_mesh_verts,
            point_batch=10000,
            res_up=4,
        )
        scene_mesh.export(f"{prefix}/scene.ply")
        return

    def test_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        idx = inputs["idx"]
        total_pixels = inputs["total_pixels"]
        prefix = f"epoch_{self.current_epoch:04d}"

        if self.opt_smpl:
            body_model_params = self.body_model_params(idx)
            smpl_scale = body_model_params["scale"]
            smpl_trans = body_model_params["transl"]
            smpl_pose = torch.cat(
                (body_model_params["global_orient"], body_model_params["body_pose"]), dim=1
            )
            smpl_shape = body_model_params["betas"]
        else:
            smpl_scale, smpl_trans, smpl_pose, smpl_shape = torch.split(
                inputs["smpl_params"], [1, 3, 72, 10], dim=1
            )

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(idx)

        smpl_outputs = self.model.smpl_server(smpl_scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs["smpl_tfs"]
        cond = {"smpl": smpl_pose[:, 3:] / np.pi}

        idx_int = idx.item()
        # create mesh
        mesh_canonical = mesh_utils.generate_mesh(
            lambda x: self.query_fg_oc(x, cond),
            self.model.smpl_server.verts_c[0],
            point_batch=10000,
            res_up=3,
        )
        mesh_canonical.export(f"{prefix}/mesh_canonical/{idx_int:04d}.ply")

        verts_deformed = self.get_deformed_mesh(mesh_canonical.vertices, cond, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(
            vertices=verts_deformed, faces=mesh_canonical.faces, process=False
        )
        mesh_deformed.export(f"{prefix}/mesh_deformed/{idx_int:04d}.ply")

        if not self.test_render_image:
            return {}

        pixel_per_batch = self.pixel_per_batch
        num_splits = torch.div(
            (total_pixels + pixel_per_batch - 1), pixel_per_batch, rounding_mode="floor"
        )
        results = []
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch, min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {
                "uv": inputs["uv"][:, indices],
                "pose": inputs["pose"],
                "P": inputs["P"],
                "C": inputs["C"],
                "intrinsics": inputs["intrinsics"],
                "smpl_params": inputs["smpl_params"],
                "smpl_scale": smpl_scale,
                "smpl_pose": smpl_pose,
                "smpl_shape": smpl_shape,
                "smpl_trans": smpl_trans,
                "idx": idx,
            }

            batch_targets = {
                "rgb_gt": targets["rgb"][:, indices].detach(),
                "normal_gt": targets["normal"][:, indices].detach(),
                "depth_gt": targets["depth"][:, indices].detach(),
                "img_size": targets["img_size"],
            }

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append(
                {
                    "rgb_pred": model_outputs["rgb"].detach(),
                    "fg_rgb_pred": model_outputs["fg_rgb"].detach(),
                    "bg_rgb_pred": model_outputs["bg_rgb"].detach(),
                    "normal_pred": model_outputs["normal"].detach(),
                    "fg_normal_pred": model_outputs["fg_normal"].detach(),
                    "bg_normal_pred": model_outputs["bg_normal"].detach(),
                    "depth_pred": model_outputs["depth"].detach(),
                    "fg_depth_pred": model_outputs["fg_depth"].detach(),
                    "bg_depth_pred": model_outputs["bg_depth"].detach(),
                    "fg_acc_map": model_outputs["fg_acc_map"].detach(),
                    "negative_entropy": model_outputs["negative_entropy"].detach(),
                    **batch_targets,
                }
            )

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_pred"][0] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)
        fg_rgb_pred = torch.cat([result["fg_rgb_pred"][0] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
        bg_rgb_pred = torch.cat([result["bg_rgb_pred"][0] for result in results], dim=0)
        bg_rgb_pred = bg_rgb_pred.reshape(*img_size, -1)
        normal_pred = torch.cat([result["normal_pred"][0] for result in results], dim=0)
        normal_pred = normal_pred.reshape(*img_size, -1)
        fg_normal_pred = torch.cat([result["fg_normal_pred"][0] for result in results], dim=0)
        fg_normal_pred = fg_normal_pred.reshape(*img_size, -1)
        bg_normal_pred = torch.cat([result["bg_normal_pred"][0] for result in results], dim=0)
        bg_normal_pred = bg_normal_pred.reshape(*img_size, -1)
        depth_pred = torch.cat([result["depth_pred"][0] for result in results], dim=0)
        depth_pred = depth_pred.reshape(*img_size, -1)
        fg_depth_pred = torch.cat([result["fg_depth_pred"][0] for result in results], dim=0)
        fg_depth_pred = fg_depth_pred.reshape(*img_size, -1)
        bg_depth_pred = torch.cat([result["bg_depth_pred"][0] for result in results], dim=0)
        bg_depth_pred = bg_depth_pred.reshape(*img_size, -1)
        fg_mask = torch.cat([result["fg_acc_map"][0] for result in results], dim=0)
        fg_mask = fg_mask.reshape(*img_size, -1)
        fg_mask = fg_mask.repeat(1, 1, 3)

        negative_entropy = torch.cat([result["negative_entropy"][0] for result in results], dim=0)
        negative_entropy = negative_entropy.reshape(*img_size, -1)
        negative_entropy = negative_entropy.repeat(1, 1, 3)

        h, w = img_size
        concat_dim = 1 if h > w else 0

        # RGB part
        rgb_gt = torch.cat([result["rgb_gt"] for result in results], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        rgb_error = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        rgb_error = self.sequential_colormap(rgb_error.cpu().numpy())[:, :, :3]
        # scale it up for better visualization
        rgb_error *= 5
        # rgb = torch.cat([rgb_gt, rgb_pred], dim=concat_dim).cpu().numpy()
        rgb = np.concatenate([rgb, rgb_error], axis=concat_dim)
        fg_rgb = torch.cat([rgb_gt, fg_rgb_pred], dim=concat_dim).cpu().numpy()
        bg_rgb = torch.cat([rgb_gt, bg_rgb_pred], dim=concat_dim).cpu().numpy()
        fg_mask = torch.cat([rgb_gt * fg_mask, fg_mask], dim=concat_dim).cpu().numpy()
        # bg_mask = torch.cat([rgb_gt * bg_mask, bg_mask], dim=concat_dim).cpu().numpy()
        negative_entropy = (
            torch.cat([rgb_gt * negative_entropy, negative_entropy], dim=concat_dim).cpu().numpy()
        )

        # Normal part
        normal_gt = torch.cat([result["normal_gt"] for result in results], dim=1).squeeze(0)
        normal_gt = normal_gt.reshape(*img_size, -1)
        normal_error = 1 - torch.sum(normal_pred * normal_gt, dim=-1)
        # map from [0, 2] to [0, 1] to align with normal space
        normal_error *= 0.5
        # for visualization
        normal_error *= 5
        normal_error = self.sequential_colormap(normal_error.cpu().numpy())[:, :, :3]
        normal = torch.cat([normal_gt, normal_pred], dim=concat_dim).cpu().numpy()
        normal = (normal + 1.0) * 0.5
        # normal = np.concatenate([normal, normal_error], axis=concat_dim)
        fg_normal = torch.cat([normal_gt, fg_normal_pred], dim=concat_dim).cpu().numpy()
        bg_normal = torch.cat([normal_gt, bg_normal_pred], dim=concat_dim).cpu().numpy()

        # Depth part
        # We use monocular prediction as GT for real sequences
        depth_gt = torch.cat([result["depth_gt"] for result in results], dim=1).squeeze(0)
        depth_gt = depth_gt.reshape(*img_size, -1)
        depth_gt = rearrange(depth_gt, "h w 1 -> 1 h w")
        depth_pred = rearrange(depth_pred, "h w 1 -> 1 h w")
        mask = torch.ones_like(depth_gt)
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_l1 = torch.abs(depth_gt - depth_pred * scale - shift).mean()
        depth_pred_aligned = depth_pred * scale + shift
        fg_depth_pred_aligned = fg_depth_pred * scale + shift
        bg_depth_pred_aligned = bg_depth_pred * scale + shift
        depth_gt = rearrange(depth_gt, "1 h w -> h w 1")
        depth_pred_aligned = rearrange(depth_pred_aligned, "1 h w -> h w 1")
        depth_pred = rearrange(depth_pred, "1 h w -> h w 1")
        depth = torch.cat([depth_gt, depth_pred_aligned], dim=concat_dim)
        vis_scale = 1 / (depth.max() - depth.min())
        vis_shift = depth.min()
        depth = (depth - vis_shift) * vis_scale
        depth = depth.cpu().numpy()
        depth_error = depth_pred - depth_gt
        depth_error = rearrange(depth_error, "h w 1-> h w")
        depth_error = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
        depth_error = self.diverging_colormap(depth_error.cpu().numpy())[:, :, :3]
        depth = rearrange(depth, "h w 1 -> h w")
        depth = self.depth_colormap(depth)[:, :, :3]
        # depth = np.concatenate([depth, depth_error], axis=concat_dim)
        fg_depth = torch.cat([depth_gt, fg_depth_pred_aligned], dim=concat_dim)
        fg_depth = (fg_depth - vis_shift) * vis_scale
        fg_depth = fg_depth.cpu().numpy()
        fg_depth = rearrange(fg_depth, "h w 1 -> h w")
        fg_depth = self.depth_colormap(fg_depth)[:, :, :3]
        bg_depth = torch.cat([depth_gt, bg_depth_pred_aligned], dim=concat_dim)
        bg_depth = (bg_depth - vis_shift) * vis_scale
        bg_depth = bg_depth.cpu().numpy()
        bg_depth = rearrange(bg_depth, "h w 1 -> h w")
        bg_depth = self.depth_colormap(bg_depth)[:, :, :3]

        # usually we don't need to save these numerical results
        if self.test_save_numerical_results:
            np.save(f"{prefix}/normal/{idx_int:04d}.npy", normal_pred.cpu().numpy())
            np.save(f"{prefix}/fg_normal/{idx_int:04d}_fg.npy", fg_normal_pred.cpu().numpy())
            np.save(f"{prefix}/bg_normal/{idx_int:04d}_bg.npy", bg_normal_pred.cpu().numpy())
            np.save(f"{prefix}/depth/{idx_int:04d}.npy", depth_pred.cpu().numpy())
            np.save(f"{prefix}/fg_depth/{idx_int:04d}_fg.npy", fg_depth_pred.cpu().numpy())
            np.save(f"{prefix}/bg_depth/{idx_int:04d}_bg.npy", bg_depth_pred.cpu().numpy())

        rgb = (rgb * 255).astype(np.uint8)
        fg_rgb = (fg_rgb * 255).astype(np.uint8)
        bg_rgb = (bg_rgb * 255).astype(np.uint8)
        # normal = (normal + 1) / 2
        normal = (normal * 255).astype(np.uint8)
        fg_normal = (fg_normal + 1) / 2
        fg_normal = (fg_normal * 255).astype(np.uint8)
        bg_normal = (bg_normal + 1) / 2
        bg_normal = (bg_normal * 255).astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        fg_depth = (fg_depth * 255).astype(np.uint8)
        bg_depth = (bg_depth * 255).astype(np.uint8)
        fg_mask = (fg_mask * 255).astype(np.uint8)
        negative_entropy = (negative_entropy * 255).astype(np.uint8)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_pred = torch.moveaxis(rgb_pred, -1, 0)[None, ...]
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        psnr = self.psnr(rgb_pred, rgb_gt)
        ssim = self.ssim(rgb_pred, rgb_gt)
        lpips = self.lpips(rgb_pred, rgb_gt)

        # normal_l1 = torch.abs(normal_gt - normal_pred).sum(dim=-1).mean()
        normal_cos = torch.sum(normal_pred * normal_gt, dim=-1).mean()
        depth_l1 = torch.abs(depth_gt - depth_pred_aligned).mean()

        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_fg/{idx_int:04d}_fg.png", fg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_bg/{idx_int:04d}_bg.png", bg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_mask/{idx_int:04d}_mask.png", fg_mask[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal/{idx_int:04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal_fg/{idx_int:04d}_fg.png", fg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal_bg/{idx_int:04d}_bg.png", bg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth/{idx_int:04d}.png", depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth_fg/{idx_int:04d}_fg.png", fg_depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth_bg/{idx_int:04d}_bg.png", bg_depth[:, :, ::-1])

        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            # "normal_l1": normal_l1,
            "normal_cos": normal_cos,
            "depth_l1": depth_l1,
        }

    def on_predict_start(self):
        prefix = f"epoch_{self.current_epoch:04d}/test/"
        Path(prefix).mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}/rgb").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_canonical").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_deformed").mkdir(parents=False, exist_ok=True)
        scene_mesh = mesh_utils.generate_mesh(
            lambda x: self.query_bg_oc(x, None),
            verts=self.scene_mesh_verts,
            point_batch=10000,
            res_up=4,
        )
        scene_mesh.export(f"{prefix}/scene.ply")
        return

    def on_predict_epoch_start(self):
        super().on_predict_epoch_start()
        self.pred_output_list = []
        return

    def predict_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        idx = inputs["idx"]
        total_pixels = inputs["total_pixels"]
        prefix = f"epoch_{self.current_epoch:04d}/test"

        # create mesh
        # use unoptimized smpl params
        smpl_scale, smpl_trans, smpl_pose, smpl_shape = torch.split(
            inputs["smpl_params"], [1, 3, 72, 10], dim=1
        )

        smpl_outputs = self.model.smpl_server(smpl_scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs["smpl_tfs"]
        cond = {"smpl": smpl_pose[:, 3:] / np.pi}

        idx_int = idx.item()
        mesh_canonical = mesh_utils.generate_mesh(
            lambda x: self.query_fg_oc(x, cond),
            self.model.smpl_server.verts_c[0],
            point_batch=10000,
            res_up=4,
        )
        mesh_canonical.export(f"{prefix}/mesh_canonical/{idx_int:04d}.ply")

        verts_deformed = self.get_deformed_mesh(mesh_canonical.vertices, cond, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(
            vertices=verts_deformed, faces=mesh_canonical.faces, process=False
        )
        mesh_deformed.export(f"{prefix}/mesh_deformed/{idx_int:04d}.ply")

        pixel_per_batch = self.pixel_per_batch
        num_splits = torch.div(
            (total_pixels + pixel_per_batch - 1), pixel_per_batch, rounding_mode="floor"
        )
        results = []
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch, min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {
                "uv": inputs["uv"][:, indices],
                "pose": inputs["pose"],
                "P": inputs["P"],
                "C": inputs["C"],
                "intrinsics": inputs["intrinsics"],
                "smpl_params": inputs["smpl_params"],
                "smpl_scale": smpl_scale,
                "smpl_pose": smpl_pose,
                "smpl_shape": smpl_shape,
                "smpl_trans": smpl_trans,
                "idx": idx,
            }

            batch_targets = {
                "rgb_gt": targets["rgb"][:, indices].detach(),
                "normal_gt": targets["normal"][:, indices].detach(),
                "depth_gt": targets["depth"][:, indices].detach(),
                "img_size": targets["img_size"],
            }

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append(
                {
                    "rgb_pred": model_outputs["rgb"].detach(),
                    "fg_rgb_pred": model_outputs["fg_rgb"].detach(),
                    "bg_rgb_pred": model_outputs["bg_rgb"].detach(),
                    "normal_pred": model_outputs["normal"].detach(),
                    "fg_normal_pred": model_outputs["fg_normal"].detach(),
                    "bg_normal_pred": model_outputs["bg_normal"].detach(),
                    "depth_pred": model_outputs["depth"].detach(),
                    "fg_depth_pred": model_outputs["fg_depth"].detach(),
                    "bg_depth_pred": model_outputs["bg_depth"].detach(),
                    "fg_acc_map": model_outputs["fg_acc_map"].detach(),
                    "negative_entropy": model_outputs["negative_entropy"].detach(),
                    **batch_targets,
                }
            )

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_pred"][0] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)
        fg_rgb_pred = torch.cat([result["fg_rgb_pred"][0] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
        bg_rgb_pred = torch.cat([result["bg_rgb_pred"][0] for result in results], dim=0)
        bg_rgb_pred = bg_rgb_pred.reshape(*img_size, -1)
        fg_mask = torch.cat([result["fg_acc_map"][0] for result in results], dim=0)
        fg_mask = fg_mask.reshape(*img_size, -1)
        fg_mask = fg_mask.repeat(1, 1, 3)
        negative_entropy = torch.cat([result["negative_entropy"][0] for result in results], dim=0)
        negative_entropy = negative_entropy.reshape(*img_size, -1)
        negative_entropy = negative_entropy.repeat(1, 1, 3)
        normal_pred = torch.cat([result["normal_pred"][0] for result in results], dim=0)
        normal_pred = normal_pred.reshape(*img_size, -1)
        fg_normal_pred = torch.cat([result["fg_normal_pred"][0] for result in results], dim=0)
        fg_normal_pred = fg_normal_pred.reshape(*img_size, -1)
        bg_normal_pred = torch.cat([result["bg_normal_pred"][0] for result in results], dim=0)
        bg_normal_pred = bg_normal_pred.reshape(*img_size, -1)
        depth_pred = torch.cat([result["depth_pred"][0] for result in results], dim=0)
        depth_pred = depth_pred.reshape(*img_size, -1)
        fg_depth_pred = torch.cat([result["fg_depth_pred"][0] for result in results], dim=0)
        fg_depth_pred = fg_depth_pred.reshape(*img_size, -1)
        bg_depth_pred = torch.cat([result["bg_depth_pred"][0] for result in results], dim=0)
        bg_depth_pred = bg_depth_pred.reshape(*img_size, -1)

        h, w = img_size
        concat_dim = 1 if h > w else 0

        rgb_gt = torch.cat([result["rgb_gt"] for result in results], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        rgb = torch.cat([rgb_gt, rgb_pred], dim=concat_dim).cpu().numpy()
        rgb_error = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        rgb_error *= 5
        rgb_error = self.sequential_colormap(rgb_error.cpu().numpy())[:, :, :3]
        rgb = np.concatenate([rgb, rgb_error], axis=concat_dim)
        fg_rgb = torch.cat([rgb_gt, fg_rgb_pred], dim=concat_dim).cpu().numpy()
        bg_rgb = torch.cat([rgb_gt, bg_rgb_pred], dim=concat_dim).cpu().numpy()
        fg_mask = torch.cat([rgb_gt * fg_mask, fg_mask], dim=concat_dim).cpu().numpy()
        negative_entropy = (
            torch.cat([rgb_gt * negative_entropy, negative_entropy], dim=concat_dim).cpu().numpy()
        )

        normal_gt = torch.cat([result["normal_gt"] for result in results], dim=1).squeeze(0)
        normal_gt = normal_gt.reshape(*img_size, -1)
        normal = torch.cat([normal_gt, normal_pred], dim=concat_dim).cpu().numpy()
        normal = (normal + 1.0) * 0.5
        normal_error = 1.0 - torch.sum(normal_pred * normal_gt, dim=-1)
        # map to [0, 1]
        normal_error *= 0.5
        # for visualization
        normal_error *= 5
        normal_error = self.sequential_colormap(normal_error.cpu().numpy())[:, :, :3]
        normal = np.concatenate([normal, normal_error], axis=concat_dim)
        fg_normal = torch.cat([normal_gt, fg_normal_pred], dim=concat_dim).cpu().numpy()
        fg_normal = (fg_normal + 1.0) * 0.5
        bg_normal = torch.cat([normal_gt, bg_normal_pred], dim=concat_dim).cpu().numpy()
        bg_normal = (bg_normal + 1.0) * 0.5

        depth_gt = torch.cat([result["depth_gt"] for result in results], dim=1).squeeze(0)
        depth_gt = depth_gt.reshape(*img_size, -1)

        np.save(f"{prefix}/normal/{idx_int:04d}.npy", normal_pred.cpu().numpy())
        np.save(f"{prefix}/normal/{idx_int:04d}_fg.npy", fg_normal_pred.cpu().numpy())
        np.save(f"{prefix}/normal/{idx_int:04d}_bg.npy", bg_normal_pred.cpu().numpy())
        np.save(f"{prefix}/depth/{idx_int:04d}.npy", depth_pred.cpu().numpy())
        np.save(f"{prefix}/depth/{idx_int:04d}_fg.npy", fg_depth_pred.cpu().numpy())
        np.save(f"{prefix}/depth/{idx_int:04d}_bg.npy", bg_depth_pred.cpu().numpy())

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_pred = torch.moveaxis(rgb_pred, -1, 0)[None, ...]
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        psnr = self.psnr(rgb_pred, rgb_gt)
        ssim = self.ssim(rgb_pred, rgb_gt)
        lpips = self.lpips(rgb_pred, rgb_gt)

        # normal_l1 = torch.abs(normal_gt - normal_pred).sum(dim=-1).mean()
        normal_cos = torch.sum(normal_pred * normal_gt, dim=-1).mean()

        depth_gt = rearrange(depth_gt, "h w 1 -> 1 h w")
        depth_pred = rearrange(depth_pred, "h w 1 -> 1 h w")
        mask = torch.ones_like(depth_gt)
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_l1 = torch.abs(depth_gt - depth_pred * scale - shift).mean()
        depth_pred_aligned = depth_pred * scale + shift
        fg_depth_pred_aligned = fg_depth_pred * scale + shift
        bg_depth_pred_aligned = bg_depth_pred * scale + shift

        depth_gt = rearrange(depth_gt, "1 h w -> h w 1")
        depth_pred = rearrange(depth_pred, "1 h w -> h w 1")
        depth_pred_aligned = rearrange(depth_pred_aligned, "1 h w -> h w 1")
        depth = torch.cat([depth_gt, depth_pred_aligned], dim=concat_dim)
        vis_scale = 1 / (depth.max() - depth.min())
        vis_shift = depth.min()
        depth = (depth - vis_shift) * vis_scale
        depth = depth.cpu().numpy()
        depth_error = depth_pred - depth_gt
        depth_error = rearrange(depth_error, "h w 1-> h w")
        depth_error = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
        depth_error = self.diverging_colormap(depth_error.cpu().numpy())[:, :, :3]
        depth = rearrange(depth, "h w 1 -> h w")
        depth = self.depth_colormap(depth)[:, :, :3]
        depth = np.concatenate([depth, depth_error], axis=concat_dim)
        fg_depth = torch.cat([depth_gt, fg_depth_pred_aligned], dim=concat_dim)
        fg_depth = (fg_depth - vis_shift) * vis_scale
        fg_depth = fg_depth.cpu().numpy()
        fg_depth = rearrange(fg_depth, "h w 1 -> h w")
        fg_depth = self.depth_colormap(fg_depth)[:, :, :3]
        bg_depth = torch.cat([depth_gt, bg_depth_pred_aligned], dim=concat_dim)
        bg_depth = (bg_depth - vis_shift) * vis_scale
        bg_depth = bg_depth.cpu().numpy()
        bg_depth = rearrange(bg_depth, "h w 1 -> h w")
        bg_depth = self.depth_colormap(bg_depth)[:, :, :3]

        rgb = (rgb * 255).astype(np.uint8)
        fg_rgb = (fg_rgb * 255).astype(np.uint8)
        bg_rgb = (bg_rgb * 255).astype(np.uint8)
        fg_mask = (fg_mask * 255).astype(np.uint8)
        negative_entropy = (negative_entropy * 255).astype(np.uint8)
        normal = (normal * 255).astype(np.uint8)
        fg_normal = (fg_normal * 255).astype(np.uint8)
        bg_normal = (bg_normal * 255).astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        fg_depth = (fg_depth * 255).astype(np.uint8)
        bg_depth = (bg_depth * 255).astype(np.uint8)

        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}_fg.png", fg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}_bg.png", bg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}_mask.png", fg_mask[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal/{idx_int:04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal/{idx_int:04d}_fg.png", fg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal/{idx_int:04d}_bg.png", bg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth/{idx_int:04d}.png", depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth/{idx_int:04d}_fg.png", fg_depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth/{idx_int:04d}_bg.png", bg_depth[:, :, ::-1])

        outputs = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "normal_cos": normal_cos,
            "depth_l1": depth_l1,
        }
        self.pred_output_list.append(outputs)
        return outputs

    def on_predict_epoch_end(self):
        prefix = f"epoch_{self.current_epoch:04d}"
        outputs = self.pred_output_list
        # outputs = outputs[0]
        with open(f"{prefix}/test_metrics.csv", "wt") as f:
            writer = csv.writer(f)
            writer.writerow(outputs[0].keys())
            for output in outputs:
                writer.writerow([value.item() for value in output.values()])
        metrics = {
            key: torch.stack([output[key] for output in outputs]).cpu().numpy()
            for key in outputs[0].keys()
        }
        with open(f"{prefix}/test_metrics_raw.pkl", "wb") as f:
            pickle.dump(metrics, f)
        return

    def get_deformed_mesh(self, verts, cond, smpl_tfs):
        verts = torch.tensor(verts, device=self.device, dtype=torch.float32)
        weights = self.model.fg_deformer.query_weights(verts, cond)
        verts_deformed = skinning(verts.unsqueeze(0), weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def on_load_checkpoint(self, checkpoint):
        self.model.mesh_v_cano = checkpoint["state_dict"]["model.mesh_v_cano"]
        self.model.mesh_f_cano = checkpoint["state_dict"]["model.mesh_f_cano"]
        self.model.mesh_face_vertices = checkpoint["state_dict"]["model.mesh_face_vertices"]
        return super().on_load_checkpoint(checkpoint)


class HSRLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eikonal_weight = cfg.eikonal_weight
        self.density_weight = cfg.density_weight
        self.normal_l1_weight = cfg.normal_l1_weight
        self.normal_cos_weight = cfg.normal_cos_weight
        self.normal_smooth_weight = cfg.normal_smooth_weight
        self.depth_weight = cfg.depth_weight
        self.off_surface_weight = cfg.off_surface_weight
        self.in_surface_weight = cfg.in_surface_weight
        self.penetration_weight = cfg.penetration_weight
        self.mask_weight = cfg.mask_weight
        self.s3im_weight = cfg.s3im_weight
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.s3im_loss = S3IM(
            s3im_kernel_size=4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=32
        )
        self.step = 0.0
        self.max_steps = cfg.max_steps
        self.penetration_loss_steps = cfg.max_steps // 2

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
        return eikonal_loss

    def get_depth_loss(self, depth_pred, depth_gt, mask):
        return self.depth_loss(depth_pred[:, :, None], depth_gt[:, :, None], mask[:, :, None])

    def get_normal_loss(self, normal_pred, normal_gt, mask):
        if not mask.any():
            return torch.zeros((), device=mask.device), torch.zeros((), device=mask.device)
        l1 = torch.abs(normal_pred[mask] - normal_gt[mask]).sum(dim=-1).mean()
        cos = (1.0 - torch.sum(normal_pred[mask] * normal_gt[mask], dim=-1)).mean()
        return l1, cos

    def get_smooth_loss(self, model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs["grad_theta"]
        g2 = model_outputs["grad_theta_nei"]

        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss = torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss

    def get_density_loss(self, acc_map):
        binary_loss = (
            -1
            * (acc_map * (acc_map + 1e-4).log() + (1 - acc_map) * (1 - acc_map + 1e-4).log()).mean()
        )
        density_loss = 2 * binary_loss
        return density_loss

    def get_off_surface_loss(self, fg_acc_map, index_off_surface):
        off_surface_loss = self.l1_loss(
            fg_acc_map[index_off_surface], torch.zeros_like(fg_acc_map[index_off_surface])
        )
        return off_surface_loss

    def get_in_surface_loss(self, fg_acc_map, index_in_surface):
        in_surface_loss = self.l1_loss(
            fg_acc_map[index_in_surface], torch.ones_like(fg_acc_map[index_in_surface])
        )
        return in_surface_loss

    def get_mask_loss(self, acc_map_fg, acc_map_bg, fg_mask):
        fg_mask_loss = self.l1_loss(acc_map_fg, fg_mask)
        bg_mask_loss = self.l1_loss(acc_map_bg, 1 - fg_mask)
        return fg_mask_loss + bg_mask_loss

    def get_penetration_loss(self, fg_sdf, bg_samples_fg_sdf, bg_sdf, fg_samples_bg_sdf):
        loss = torch.zeros((), device=fg_sdf.device)
        fg_samples_penetration_mask = torch.logical_and(fg_sdf < 0, fg_samples_bg_sdf < 0)
        if fg_samples_penetration_mask.any():
            loss += (
                fg_sdf[fg_samples_penetration_mask] * fg_samples_bg_sdf[fg_samples_penetration_mask]
            ).mean()
        bg_samples_penetration_mask = torch.logical_and(bg_sdf < 0, bg_samples_fg_sdf < 0)
        if bg_samples_penetration_mask.any():
            loss += (
                bg_sdf[bg_samples_penetration_mask] * bg_samples_fg_sdf[bg_samples_penetration_mask]
            ).mean()
        return loss

    def forward(self, model_outputs, ground_truth):
        rgb_loss = self.get_rgb_loss(model_outputs["rgb"], ground_truth["rgb"])
        fg_eikonal_loss = self.eikonal_weight * self.get_eikonal_loss(
            model_outputs["fg_grad_theta"]
        )
        bg_eikonal_loss = self.eikonal_weight * self.get_eikonal_loss(
            model_outputs["bg_grad_theta"]
        )
        if self.density_weight > 0:
            density_loss = self.density_weight * (
                self.get_density_loss(model_outputs["fg_acc_map"])
                + self.get_density_loss(model_outputs["bg_acc_map"])
            )
        else:
            density_loss = torch.zeros((), device=rgb_loss.device)
        off_surface_weight = self.off_surface_weight * (1 + 10 * self.step / self.max_steps)
        if off_surface_weight > 0:
            off_surface_loss = off_surface_weight * self.get_off_surface_loss(
                model_outputs["fg_acc_map"], model_outputs["index_off_surface"]
            )
        else:
            off_surface_loss = torch.zeros((), device=rgb_loss.device)
        in_surface_weight = self.in_surface_weight * (1 - self.step / self.max_steps)
        if in_surface_weight > 0:
            in_surface_loss = in_surface_weight * self.get_in_surface_loss(
                model_outputs["fg_acc_map"], model_outputs["index_in_surface"]
            )
        else:
            in_surface_loss = torch.zeros((), device=rgb_loss.device)

        # b x n x m
        bg_sdf = model_outputs["bg_sdf"]
        bg_sdf_mask = torch.logical_and((bg_sdf < 0.0).any(dim=2), (bg_sdf > 0.0).any(dim=2))
        fg_sdf = model_outputs["fg_sdf"]
        fg_sdf_mask = torch.logical_and((fg_sdf < 0.0).any(dim=2), (fg_sdf > 0.0).any(dim=2))
        sdf_mask = torch.logical_or(bg_sdf_mask, fg_sdf_mask)
        human_mask = ground_truth["human_mask"]
        non_human_mask = torch.logical_not(human_mask)
        mask = torch.logical_and(sdf_mask, non_human_mask)
        depth_weight = self.depth_weight  # * (1 - min(1, 2 * self.step / self.max_steps))
        if self.step < self.max_steps / 100:
            depth_weight = 0.0
        else:
            depth_weight = depth_weight
        if depth_weight > 0:
            depth_loss = depth_weight * self.get_depth_loss(
                model_outputs["depth"], ground_truth["depth"], mask
            )
        else:
            depth_loss = torch.zeros((), device=rgb_loss.device)
        normal_l1_weight = self.normal_l1_weight  # * (1 - min(1, 2 * self.step / self.max_steps))
        normal_cos_weight = self.normal_cos_weight  # * (1 - min(1, 2 * self.step / self.max_steps))
        if self.normal_l1_weight > 0 or self.normal_cos_weight > 0:
            normal_l1_loss, normal_cos_loss = self.get_normal_loss(
                model_outputs["normal"], ground_truth["normal"], mask
            )
            normal_l1_loss = normal_l1_weight * normal_l1_loss
            normal_cos_loss = normal_cos_weight * normal_cos_loss
        else:
            normal_l1_loss = torch.zeros((), device=rgb_loss.device)
            normal_cos_loss = torch.zeros((), device=rgb_loss.device)
        if self.step > self.penetration_loss_steps:
            penetration_loss = self.penetration_weight * self.get_penetration_loss(
                fg_sdf,
                model_outputs["bg_samples_fg_sdf"],
                bg_sdf,
                model_outputs["fg_samples_bg_sdf"],
            )
        else:
            penetration_loss = torch.zeros((), device=rgb_loss.device)

        if self.mask_weight > 0:
            mask_loss = self.mask_weight * self.get_mask_loss(
                model_outputs["acc_map_fg"], model_outputs["acc_map_bg"], ground_truth["human_mask"]
            )
        else:
            mask_loss = torch.zeros((), device=rgb_loss.device)

        if self.s3im_weight > 0:
            s3im_loss = self.s3im_weight * self.s3im_loss(model_outputs["rgb"], ground_truth["rgb"])
        else:
            s3im_loss = torch.zeros((), device=rgb_loss.device)

        loss = (
            rgb_loss
            + fg_eikonal_loss
            + bg_eikonal_loss
            + mask_loss
            + density_loss
            + off_surface_loss
            + in_surface_loss
            + depth_loss
            + normal_l1_loss
            + normal_cos_loss
            + penetration_loss
            + s3im_loss
        )
        self.step = self.step + 1.0
        return {
            "train/loss": loss,
            "train/rgb": rgb_loss,
            "train/s3im": s3im_loss,
            "train/mask": mask_loss,
            "train/depth": depth_loss,
            "train/normal_cos": normal_cos_loss,
            "train/penetration": penetration_loss,
            "train/fg_eikonal": fg_eikonal_loss,
            "train/bg_eikonal": bg_eikonal_loss,
            "train/density_reg": density_loss,
            "train/off_surface": off_surface_loss,
            "train/in_surface": in_surface_loss,
            "train/normal_l1": normal_l1_loss,
        }
