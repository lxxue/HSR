import abc

import torch

from hsr.utils import render_utils


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class BBoxSampler(RaySampler):
    def __init__(self, N_samples, near=0.0, far=6.0, **kwargs):
        super().__init__(near, far)  # indeed, we need to set the bounds for every iteration
        self.perturb = 1
        self.N_samples = N_samples

    # https://github.com/yenchenlin/nerf-pytorch
    def get_wsampling_points(self, ray_o, ray_d, near, far, is_training):
        # calculate the steps for each ray
        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).to(near)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals

        if self.perturb > 0.0 and is_training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand
        # pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        # return pts, z_vals
        return z_vals

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, smpl_mesh, is_training):
        self.near, self.far, padded_ray_dirs = render_utils.get_bbox_intersection(
            cam_loc, ray_dirs, smpl_mesh
        )
        if padded_ray_dirs is None:
            z_vals = self.get_wsampling_points(cam_loc, ray_dirs, self.near, self.far, is_training)
        else:
            z_vals = self.get_wsampling_points(
                cam_loc, padded_ray_dirs, self.near, self.far, is_training
            )
        return z_vals


class UniformSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        take_sphere_intersection=False,
        far=-1.0,
    ):
        super().__init__(
            near, 2.0 * scene_bounding_sphere if far == -1 else far
        )  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = (
                self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device),
                self.far * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device),
            )
        else:
            sphere_intersections = render_utils.get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )
            near = self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
            far = sphere_intersections[:, 1:]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=ray_dirs.device)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)

            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class ErrorBoundSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        inverse_sphere_bg=False,
        N_samples_inverse_sphere=0,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            near,
            N_samples_eval,
            take_sphere_intersection=inverse_sphere_bg,
        )

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            N_samples_inverse_sphere = 32
            self.inverse_sphere_sampler = UniformSampler(
                1.0, 0.0, N_samples_inverse_sphere, False, far=1.0
            )

    @torch.no_grad()
    def get_z_vals(
        self,
        ray_dirs,
        cam_loc,
        model,
        cond,
        smpl_tfs,
        eval_mode,
        smpl_verts,
    ):
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists**2.0).sum(-1)
        beta = torch.sqrt(bound + 1e-8)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            # Calculating the SDF only for the new sampled points
            model.implicit_network.eval()
            with torch.no_grad():
                samples_sdf = model.sdf_func_with_smpl_deformer(
                    points_flat,
                    cond,
                    smpl_tfs,
                    smpl_verts=smpl_verts,
                )[0]
            model.implicit_network.train()
            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    -1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros((z_vals.shape[0], z_vals.shape[1] - 1), device=z_vals.device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask] + 1e-8)) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10], device=dists.device)
                    .unsqueeze(0)
                    .repeat(dists.shape[0], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [
                    torch.zeros((dists.shape[0], 1), device=dists.device),
                    free_energy[:, :-1],
                ],
                dim=-1,
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[:, :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = (
                    torch.linspace(0.0, 1.0, steps=N, device=dists.device)
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N], device=cdf.device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        near, far = (
            self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device),
            self.far * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device),
        )
        if self.inverse_sphere_bg:  # if inverse sphere then need to add the far sphere intersection
            far = render_utils.get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=z_vals.device)
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                dists * density[:, :-1],
            ],
            dim=-1,
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1]
        )

        return bound_opacity.max(-1)[0]


def compute_bbox_nf(rays_o, rays_d, bbox_min, bbox_max, R):
    x1 = bbox_min[0]
    x2 = bbox_max[0]
    y1 = bbox_min[1]
    y2 = bbox_max[1]
    z1 = bbox_min[2]
    z2 = bbox_max[2]

    device = rays_o.device
    delta = torch.tensor([1e-10], device=device)
    # for numerical stability when computing intersections with the bbox
    rays_dx = torch.sign(rays_d[:, 0]) * torch.max(torch.abs(rays_d[:, 0]), delta)
    rays_dy = torch.sign(rays_d[:, 1]) * torch.max(torch.abs(rays_d[:, 1]), delta)
    rays_dz = torch.sign(rays_d[:, 2]) * torch.max(torch.abs(rays_d[:, 2]), delta)

    # a dirty fix for rays parallel to one axis, should be rare but might occur in high_res data
    rays_dx[rays_d[:, 0] == 0] = delta
    rays_dy[rays_d[:, 1] == 0] = delta
    rays_dz[rays_d[:, 2] == 0] = delta

    inverse_rays_dx = 1.0 / rays_dx
    inverse_rays_dy = 1.0 / rays_dy
    inverse_rays_dz = 1.0 / rays_dz

    assert not (torch.isinf(inverse_rays_dx).any() or torch.isnan(inverse_rays_dx).any())
    assert not (torch.isinf(inverse_rays_dy).any() or torch.isnan(inverse_rays_dy).any())
    assert not (torch.isinf(inverse_rays_dz).any() or torch.isnan(inverse_rays_dz).any())

    bound_x1 = (x1 - rays_o[:, 0]) * inverse_rays_dx
    bound_x2 = (x2 - rays_o[:, 0]) * inverse_rays_dx
    bound_y1 = (y1 - rays_o[:, 1]) * inverse_rays_dy
    bound_y2 = (y2 - rays_o[:, 1]) * inverse_rays_dy
    bound_z1 = (z1 - rays_o[:, 2]) * inverse_rays_dz
    bound_z2 = (z2 - rays_o[:, 2]) * inverse_rays_dz

    bound_x_near = torch.minimum(bound_x1, bound_x2)
    bound_x_far = torch.maximum(bound_x1, bound_x2)
    bound_y_near = torch.minimum(bound_y1, bound_y2)
    bound_y_far = torch.maximum(bound_y1, bound_y2)
    bound_z_near = torch.minimum(bound_z1, bound_z2)
    bound_z_far = torch.maximum(bound_z1, bound_z2)

    near = torch.maximum(torch.maximum(bound_x_near, bound_y_near), bound_z_near)
    far = torch.minimum(torch.minimum(bound_x_far, bound_y_far), bound_z_far)

    # in case a ray miss the AABB
    diff = far - near
    miss = diff < 0
    # near[miss] = 0.0
    # far[miss] = 2.0 * R
    near[miss] = far[miss]

    return near, far


class BBoxRaySampler(RaySampler):
    def __init__(self, N_samples, near, human_bound_sphere):
        super().__init__(near, far=2 * human_bound_sphere)
        self.N_samples = N_samples
        self.human_bound_sphere = human_bound_sphere
        self.near_ = torch.tensor([0], device="cuda:0")
        self.far_ = torch.tensor([2 * human_bound_sphere], device="cuda:0")

    @torch.no_grad()
    def get_z_vals(self, ray_dir, cam_loc, bbox_min, bbox_max, is_training):
        near, far = compute_bbox_nf(cam_loc, ray_dir, bbox_min, bbox_max, self.human_bound_sphere)
        near = torch.maximum(near, self.near_)
        near = torch.minimum(near, self.far_)
        far = torch.maximum(far, self.near_)
        far = torch.minimum(far, self.far_)
        # calculate the steps for each ray
        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).to(near)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals

        if is_training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        return near, far, z_vals


class ErrorBoundSamplerWithDeformerAndBBox(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.N_samples_extra = N_samples_extra
        self.bbox_sampler = BBoxRaySampler(N_samples_eval, near, scene_bounding_sphere)

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

    @torch.no_grad()
    def get_z_vals(
        self,
        ray_dirs,
        cam_loc,
        model,
        cond,
        smpl_tfs,
        smpl_verts,
    ):
        beta0 = model.fg_density.get_beta().detach()

        # Start with uniform sampling
        with torch.no_grad():
            bbox_min = smpl_verts.min(dim=1)[0][0]
            bbox_max = smpl_verts.max(dim=1)[0][0]
            bbox_size = bbox_max - bbox_min
            bbox_min -= bbox_size * 0.2
            bbox_max += bbox_size * 0.2
            near, far, z_vals = self.bbox_sampler.get_z_vals(
                ray_dirs,
                cam_loc,
                bbox_min,
                bbox_max,
                model.training,
            )
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists**2.0).sum(-1)
        beta = torch.sqrt(bound + 1e-8)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            # Calculating the SDF only for the new sampled points
            model.fg_implicit_network.eval()
            with torch.no_grad():
                samples_sdf = model.sdf_func_with_smpl_deformer(
                    points_flat,
                    cond,
                    smpl_tfs,
                    smpl_verts=smpl_verts,
                )[0]
            model.fg_implicit_network.train()
            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    -1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros((z_vals.shape[0], z_vals.shape[1] - 1), device=z_vals.device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask] + 1e-8)) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.fg_density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10], device=dists.device)
                    .unsqueeze(0)
                    .repeat(dists.shape[0], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [
                    torch.zeros((dists.shape[0], 1), device=dists.device),
                    free_energy[:, :-1],
                ],
                dim=-1,
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[:, :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = (
                    torch.linspace(0.0, 1.0, steps=N, device=dists.device)
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N], device=cdf.device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        # near = self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
        # # far = self.far * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
        # far = render_utils.get_sphere_intersections(
        #     cam_loc, ray_dirs, r=self.scene_bounding_sphere
        # )[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near[:, None], far[:, None], z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near[:, None], far[:, None]], -1)

        # N_samples + N_samples_extra + 2
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        return z_vals

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.fg_density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                dists * density[:, :-1],
            ],
            dim=-1,
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1]
        )

        return bound_opacity.max(-1)[0]


# No inverse sphere samples for NeRF++
class ErrorBoundSamplerWithDeformer(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.N_samples_extra = N_samples_extra
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            near,
            N_samples_eval,
            take_sphere_intersection=True,
        )

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

    @torch.no_grad()
    def get_z_vals(
        self,
        ray_dirs,
        cam_loc,
        model,
        cond,
        smpl_tfs,
        smpl_verts,
    ):
        beta0 = model.fg_density.get_beta().detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists**2.0).sum(-1)
        beta = torch.sqrt(bound + 1e-8)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            # Calculating the SDF only for the new sampled points
            model.fg_implicit_network.eval()
            with torch.no_grad():
                samples_sdf = model.sdf_func_with_smpl_deformer(
                    points_flat,
                    cond,
                    smpl_tfs,
                    smpl_verts=smpl_verts,
                )[0]
            model.fg_implicit_network.train()
            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    -1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros((z_vals.shape[0], z_vals.shape[1] - 1), device=z_vals.device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask] + 1e-8)) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.fg_density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10], device=dists.device)
                    .unsqueeze(0)
                    .repeat(dists.shape[0], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [
                    torch.zeros((dists.shape[0], 1), device=dists.device),
                    free_energy[:, :-1],
                ],
                dim=-1,
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[:, :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = (
                    torch.linspace(0.0, 1.0, steps=N, device=dists.device)
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N], device=cdf.device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        near = self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
        # far = self.far * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
        far = render_utils.get_sphere_intersections(
            cam_loc, ray_dirs, r=self.scene_bounding_sphere
        )[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        # N_samples + N_samples_extra + 2
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        return z_vals

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.fg_density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                dists * density[:, :-1],
            ],
            dim=-1,
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1]
        )

        return bound_opacity.max(-1)[0]


class ErrorBoundSamplerWithoutDeformer(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            near,
            N_samples_eval,
            take_sphere_intersection=True,
        )

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model):
        beta0 = model.bg_density.get_beta().detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists**2.0).sum(-1)
        beta = torch.sqrt(bound + 1e-8)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            # Calculating the SDF only for the new sampled points
            model.bg_implicit_network.eval()
            with torch.no_grad():
                samples_sdf = model.sdf_func_without_smpl_deformer(points_flat, None)[0]
            model.bg_implicit_network.train()
            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    -1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros((z_vals.shape[0], z_vals.shape[1] - 1), device=z_vals.device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask] + 1e-8)) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.bg_density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10], device=dists.device)
                    .unsqueeze(0)
                    .repeat(dists.shape[0], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [
                    torch.zeros((dists.shape[0], 1), device=dists.device),
                    free_energy[:, :-1],
                ],
                dim=-1,
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[:, :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = (
                    torch.linspace(0.0, 1.0, steps=N, device=dists.device)
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N], device=cdf.device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        near = self.near * torch.ones((ray_dirs.shape[0], 1), device=ray_dirs.device)
        far = render_utils.get_sphere_intersections(
            cam_loc, ray_dirs, r=self.scene_bounding_sphere
        )[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=z_vals.device)
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.bg_density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                dists * density[:, :-1],
            ],
            dim=-1,
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1]
        )

        return bound_opacity.max(-1)[0]
