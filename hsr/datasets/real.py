import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from hsr.utils import sample_utils


class RealDataset(Dataset):
    def __init__(self, cfg):
        root = Path(cfg.data_dir)
        training_indices = list(range(cfg.start_frame, cfg.end_frame + 1, cfg.skip_step))
        if cfg.exclude_frames is not None:
            for i in cfg.exclude_frames:
                training_indices.remove(i)
        self.training_indices = np.array(training_indices)

        # images
        img_dir = root / "image"
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
        img_paths = [img_paths[i] for i in self.training_indices]
        # depth
        depth_dir = root / "depth"
        depth_paths = sorted(glob.glob(f"{depth_dir}/*.npy"))
        depth_paths = [depth_paths[i] for i in self.training_indices]
        # normal
        normal_dir = root / "normal"
        normal_paths = sorted(glob.glob(f"{normal_dir}/*.npy"))
        normal_paths = [normal_paths[i] for i in self.training_indices]
        # sam_mask
        sam_mask_dir = root / "sam_mask"
        sam_paths = sorted(glob.glob(f"{sam_mask_dir}/*.png"))
        sam_paths = [sam_paths[i] for i in self.training_indices]

        self.num_imgs = len(img_paths)
        assert self.num_imgs == len(depth_paths) == len(normal_paths) == len(sam_paths)

        shape = np.load(root / "mean_shape.npy")
        # scale factor for the SMPL model to be consistent with scene scale
        self.scale = shape[0]
        self.shape = shape[1:]
        self.poses = np.load(root / "poses.npy")[self.training_indices]
        self.trans = np.load(root / "normalize_trans.npy")[self.training_indices]
        assert self.num_imgs == self.poses.shape[0]
        assert self.num_imgs == self.trans.shape[0]

        self.P, self.C = [], []
        camera_dict = np.load(root / "cameras_normalize.npz")
        scale_mats = [camera_dict[f"scale_mat_{idx}"] for idx in self.training_indices]
        world_mats = [camera_dict[f"world_mat_{idx}"] for idx in self.training_indices]

        # do another resize for fast rendering, typically during evaluation and test
        img_resize_factor = cfg.img_resize_factor
        assert img_resize_factor in [1, 2, 4]
        img_size = cv2.imread(img_paths[0]).shape[:2]
        target_img_size = (img_size[0] // img_resize_factor, img_size[1] // img_resize_factor)
        self.img_size = target_img_size

        # assume all images share the same scale
        self.scale *= 1 / scale_mats[0][0, 0]
        intrinsic = np.load(root / "intrinsic.npy")
        intrinsic[0, 0] *= 1 / img_resize_factor
        intrinsic[1, 1] *= 1 / img_resize_factor
        intrinsic[0, 2] *= 1 / img_resize_factor
        intrinsic[1, 2] *= 1 / img_resize_factor
        c2ws = np.load(root / "c2ws.npy")
        c2ws = c2ws[self.training_indices]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat, c2w in zip(scale_mats, world_mats, c2ws):
            P = world_mat @ scale_mat
            self.P.append(P)
            C = c2w[:3, 3]
            self.C.append(C)
            self.intrinsics_all.append(torch.from_numpy(intrinsic))
            self.pose_all.append(torch.from_numpy(c2w))

        assert self.num_imgs == len(self.P) == len(self.C)
        assert self.num_imgs == len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = cfg.num_sample
        if "sampling_strategy" in cfg:
            self.sampling_strategy = cfg.sampling_strategy
        if "bbox_sampling_ratio" in cfg:
            self.bbox_sampling_ratio = cfg.bbox_sampling_ratio

        self.imgs = np.zeros(
            (self.num_imgs, target_img_size[0], target_img_size[1], 3), dtype=np.float32
        )
        self.depths = np.zeros(
            (self.num_imgs, target_img_size[0], target_img_size[1]), dtype=np.float32
        )
        self.normals = np.zeros(
            (self.num_imgs, target_img_size[0], target_img_size[1], 3), dtype=np.float32
        )
        self.sam_masks = np.zeros(
            (self.num_imgs, target_img_size[0], target_img_size[1]), dtype=np.float32
        )
        for i in range(self.num_imgs):
            img = cv2.imread(img_paths[i])
            img = cv2.resize(img, (target_img_size[1], target_img_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = img / 255.0
            self.imgs[i] = img

            depth = np.load(depth_paths[i])
            depth = cv2.resize(depth, (target_img_size[1], target_img_size[0]))
            self.depths[i] = depth

            normal = np.load(normal_paths[i])
            normal = cv2.resize(normal, (target_img_size[1], target_img_size[0]))
            # for visualization normal is mapped to [0,1] by (normal + 1) / 2
            normal = normal * 2.0 - 1.0
            # do an additional normalization just in case
            # normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
            self.normals[i] = normal

            sam_mask = cv2.imread(sam_paths[i])
            sam_mask = cv2.resize(sam_mask, (target_img_size[1], target_img_size[0]))
            sam_mask = sam_mask[:, :, 0] / 255.0
            self.sam_masks[i] = sam_mask

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img = self.imgs[idx]
        depth = self.depths[idx]
        normal = self.normals[idx]
        sam_mask = self.sam_masks[idx]

        img_size = img.shape[:2]

        uv = np.mgrid[: img_size[0], : img_size[1]]
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86], dtype=torch.float32)
        smpl_params[0] = torch.tensor([self.scale])
        smpl_params[1:4] = torch.from_numpy(self.trans[idx])
        smpl_params[4:76] = torch.from_numpy(self.poses[idx])
        smpl_params[76:] = torch.from_numpy(self.shape)

        num_sample = self.num_sample
        if num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "human_mask": sam_mask,
                "depth": depth,
                "normal": normal,
            }
            index_outside = None
            if self.sampling_strategy == "uniform":
                samples = sample_utils.uniform_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "uniform_continuous":
                samples = sample_utils.uniform_sampling_continuous(data, img_size, num_sample)
            elif self.sampling_strategy == "weighted":
                samples, index_outside = sample_utils.weighted_sampling(
                    data, img_size, num_sample, self.bbox_sampling_ratio
                )
            elif self.sampling_strategy == "bg_only":
                samples, index_outside = sample_utils.bg_only_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "bg_only_patch":
                samples, index_outside = sample_utils.bg_only_patch_sampling(
                    data, img_size, num_sample
                )
            elif self.sampling_strategy == "fg_only":
                samples, index_outside = sample_utils.fg_only_sampling(data, img_size, num_sample)
            else:
                raise ValueError("Unknown sampling strategy")
            inputs = {
                "uv": samples["uv"],
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "index_outside": index_outside,
                "idx": idx,
            }
            images = {
                "rgb": samples["rgb"],
                "depth": samples["depth"],
                "normal": samples["normal"],
                "human_mask": samples["human_mask"],
            }
        else:
            inputs = {
                "uv": uv.reshape(-1, 2),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "idx": idx,
            }
            depth = rearrange(depth, "h w -> (h w) 1")
            normal = rearrange(normal, "h w c -> (h w) c")
            images = {
                "rgb": img.reshape(-1, 3),
                "img_size": img_size,
                "depth": depth,
                "normal": normal,
                "human_mask": sam_mask,
            }
        return inputs, images


class RealValDataset(Dataset):
    def __init__(self, cfg):
        self.dataset = RealDataset(cfg)
        self.img_size = self.dataset.img_size
        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = cfg.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = np.random.randint(0, len(self.dataset))
        inputs, targets = self.dataset[image_id]
        inputs["total_pixels"] = self.total_pixels
        return inputs, targets


class RealTestDataset(Dataset):
    def __init__(self, cfg):
        self.dataset = RealDataset(cfg)
        self.img_size = self.dataset.img_size
        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = cfg.pixel_per_batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, targets = self.dataset[idx]
        inputs["total_pixels"] = self.total_pixels
        return inputs, targets
