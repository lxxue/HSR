import torch
import torch.nn.functional as F
from pytorch3d import ops

from hsr.smpl.smpl_server import SMPLServer


class SMPLDeformer(torch.nn.Module):
    def __init__(self, max_dist=0.05, K=5, gender=None, betas=None, scale=1.0):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.smpl = SMPLServer(gender=gender, scale=scale)
        smpl_params_canoical = self.smpl.param_canonical.clone()
        smpl_params_canoical[:, 76:] = torch.tensor(betas, device=self.smpl.param_canonical.device)
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(
            smpl_params_canoical, [1, 3, 72, 10], dim=1
        )
        smpl_output = self.smpl(cano_scale, cano_transl, cano_thetas, cano_betas)
        self.register_buffer("smpl_verts", smpl_output["smpl_verts"])
        self.register_buffer("smpl_weights", smpl_output["smpl_weights"])

    def forward(self, x, smpl_tfs, return_weights=True, inverse=False, smpl_verts=None):
        if smpl_verts is None:
            weights, outlier_mask = self.query_skinning_weights_smpl_multi(
                x[None], smpl_verts=self.smpl_verts[0], smpl_weights=self.smpl_weights
            )
        else:
            weights, outlier_mask = self.query_skinning_weights_smpl_multi(
                x[None], smpl_verts=smpl_verts[0], smpl_weights=self.smpl_weights
            )
        if return_weights:
            return weights
        x_transformed = skinning(x.unsqueeze(0), weights, smpl_tfs, inverse=inverse)

        return x_transformed.squeeze(0), outlier_mask

    def forward_skinning(self, xc, cond, smpl_tfs):
        weights, outlier_mask = self.query_skinning_weights_smpl_multi(
            xc, smpl_verts=self.smpl_verts[0], smpl_weights=self.smpl_weights
        )
        x_transformed = skinning(xc, weights, smpl_tfs, inverse=False)

        return x_transformed

    def query_skinning_weights_smpl_multi(self, pts, smpl_verts, smpl_weights):
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts, smpl_verts.unsqueeze(0), K=self.K, return_nn=True
        )
        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch + 1e-8)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)
        index_batch = index_batch[0]
        weights = smpl_weights[:, index_batch, :]
        weights = torch.sum(weights * weights_conf.unsqueeze(-1), dim=-2).detach()

        outlier_mask = (distance_batch[..., 0] > 0.1)[0]
        return weights, outlier_mask

    def query_skinning_weights_smpl(self, pts, smpl_verts, smpl_weights):
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts, smpl_verts.unsqueeze(0), K=self.K, return_nn=True
        )

        neighbor_points = neighbor_points[0]
        distance_batch = torch.sqrt(distance_batch[0] + 1e-8)
        index_batch = index_batch[0, :, 0]

        outlier_mask = distance_batch > 0.1

        weights = smpl_weights[:, index_batch, :]
        return weights, outlier_mask

    def query_weights(self, xc, cond):
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse=False):
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc[0], cond)
        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum("bpn,bnij->bpij", w.double(), tfs.double())
            p_h = torch.einsum("bpij,bpj->bpi", tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum(
                "bpn, bnij, bpj->bpi", w.double(), tfs.double(), p_h.double()
            ).float()

        return p_h[:, :, :3]


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
    return x_h[:, :, :3]
