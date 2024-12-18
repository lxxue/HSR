from torch.utils.data import DataLoader

from hsr.datasets.real import RealDataset, RealTestDataset, RealValDataset


def find_dataset_classes(name):
    mapping = {
        "real": (RealDataset, RealValDataset, RealTestDataset),
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}")
    return cls


def create_dataloaders(cfg):
    train_cls, val_cls, test_cls = find_dataset_classes(cfg.name)
    train_dloader = create_dataloader(cfg.train, train_cls)
    val_dloader = create_dataloader(cfg.val, val_cls)
    test_dloader = create_dataloader(cfg.test, test_cls)
    return train_dloader, val_dloader, test_dloader


def create_dataloader(cfg, dset_cls):
    dset = dset_cls(cfg)
    dloader = DataLoader(
        dset,
        batch_size=cfg.batch_size,
        drop_last=cfg.drop_last,
        shuffle=cfg.shuffle,
        num_workers=cfg.worker,
        pin_memory=True,
    )
    return dloader
