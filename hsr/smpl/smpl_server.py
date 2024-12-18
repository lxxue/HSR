from pathlib import Path

import numpy as np
import torch

from hsr.smpl.body_models import SMPL


class SMPLServer(torch.nn.Module):
    def __init__(self, gender="neutral", betas=None, v_template=None, scale=1.0):
        super().__init__()
        model_path = Path(__file__).parents[1] / "checkpoints" / "smpl_body_models"
        self.smpl = SMPL(
            model_path=str(model_path),
            gender=gender,
            batch_size=1,
            use_hands=False,
            use_feet_keypoints=False,
            dtype=torch.float32,
        )

        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        self.faces = self.smpl.faces
        for i in range(24):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template is not None:
            self.register_buffer("v_template", torch.tensor(v_template))
        else:
            self.register_buffer("v_template", None)

        if betas is not None:
            self.register_buffer("betas", torch.tensor(betas))
        else:
            self.register_buffer("betas", None)

        # define the canonical pose
        param_canonical = torch.zeros((1, 86), dtype=torch.float32)
        param_canonical[0, 0] = 1.0
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if self.betas is not None and self.v_template is None:
            param_canonical[0, -10:] = self.betas
        self.register_buffer("param_canonical", param_canonical)

        output = self.forward(
            *torch.split(self.param_canonical, [1, 3, 72, 10], dim=1), absolute=True
        )

        self.register_buffer("verts_c", output["smpl_verts"])
        self.register_buffer("joints_c", output["smpl_jnts"])
        self.register_buffer("tfs_c_inv", output["smpl_tfs"].squeeze(0).inverse())

    def forward(self, scale, transl, thetas, betas, absolute=False):
        """return SMPL output from params
        Args:
            smpl_params : smpl parameters. shape: [B, 86]. [0-scale,1:4-trans, 4:76-thetas,76:86-betas]
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical.
        Returns:
            smpl_verts: vertices. shape: [B, 6893. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        output = {}
        # scale, transl, thetas, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        smpl_output = self.smpl.forward(
            betas=betas,
            transl=torch.zeros_like(transl),
            body_pose=thetas[:, 3:],
            global_orient=thetas[:, :3],
            return_verts=True,
            return_full_pose=True,
            v_template=self.v_template,
        )

        verts = smpl_output.vertices.clone()
        output["smpl_verts"] = verts * scale.unsqueeze(1) + transl.unsqueeze(1) * scale.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output["smpl_jnts"] = joints * scale.unsqueeze(1) + transl.unsqueeze(1) * scale.unsqueeze(1)

        # tf_mats = smpl_output.T.clone()
        # tf_mats[:, :, :3, :] = tf_mats[:, :, :3, :] * scale.unsqueeze(1).unsqueeze(1)
        # tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl.unsqueeze(
        #     1
        # ) * scale.unsqueeze(1)
        tf_mats = smpl_output.T.clone()[:, :, :3, :]
        tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl.unsqueeze(1)
        tf_mats = tf_mats * scale.unsqueeze(1).unsqueeze(1)
        # tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl.unsqueeze(
        #     1
        # ) * scale.unsqueeze(1)
        last_row = torch.tensor([0, 0, 0, 1], dtype=tf_mats.dtype, device=tf_mats.device)
        # last_row = last_row.view(1, 1, 1, 4).expand(-1, tf_mats.shape[1], -1, -1)
        last_row = last_row.view(1, 1, 1, 4).expand(tf_mats.shape[0], tf_mats.shape[1], -1, -1)
        tf_mats = torch.cat([tf_mats, last_row], dim=2)

        if not absolute:
            tf_mats = torch.einsum("bnij,njk->bnik", tf_mats, self.tfs_c_inv)

        output["smpl_tfs"] = tf_mats
        output["smpl_weights"] = smpl_output.weights
        return output
