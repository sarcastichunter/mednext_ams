import json
import os
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
)
from monai.data import Dataset as MonaiDataset


def _load_dataset_json(dataset_dir: str) -> Dict:
    dataset_json_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_json_path, "r") as f:
        return json.load(f)


def _load_splits(dataset_dir: str) -> List[Dict]:
    splits_path = os.path.join(dataset_dir, "splits_final.json")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"splits_final.json not found in {dataset_dir}")
    with open(splits_path, "r") as f:
        return json.load(f)


def _get_case_ids(dataset_dir: str) -> List[str]:
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    case_ids = sorted(
        [f.split("_0000.nii.gz")[0] for f in os.listdir(images_tr_dir) if f.endswith("_0000.nii.gz")]
    )
    return case_ids


def _build_file_lists(dataset_dir: str, case_ids: List[str]) -> List[Dict]:
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")

    data = []
    for cid in case_ids:
        img_path = os.path.join(images_tr_dir, f"{cid}_0000.nii.gz")
        lbl_path = os.path.join(labels_tr_dir, f"{cid}.nii.gz")
        if not (os.path.isfile(img_path) and os.path.isfile(lbl_path)):
            continue
        data.append({"image": img_path, "label": lbl_path})
    return data


def get_transforms(patch_size: Tuple[int, int, int] = (128, 128, 128), training: bool = True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # prilagodi okno po potrebi (npr. CTA HU range)
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    if training:
        aug = [
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        ]
        return Compose(base + aug)
    else:
        # za validacijo samo center crop (determinističen)
        val = [
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        ]
        return Compose(base + val)


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    split_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Ustvari train in val DataLoaderja za nnU-Net v2-style dataset.
    """

    _ = _load_dataset_json(dataset_dir)  # trenutno samo sanity check, lahko kasneje uporabiva info
    splits = _load_splits(dataset_dir)
    case_ids_all = _get_case_ids(dataset_dir)

    if split_id >= len(splits):
        raise ValueError(f"Requested split_id {split_id}, but only {len(splits)} splits available.")

    split = splits[split_id]

    # če so v splitu že case ID-ji (stringi), jih vzamemo direktno
    if isinstance(split["train"][0], str):
        train_ids = split["train"]
        val_ids = split["val"]
    else:
        #nnU-Net stil: indeksi -> case_ids_all
        train_ids = [case_ids_all[i] for i in split["train"]]
        val_ids = [case_ids_all[i] for i in split["val"]]


    train_data = _build_file_lists(dataset_dir, train_ids)
    val_data = _build_file_lists(dataset_dir, val_ids)

    train_transforms = get_transforms(patch_size=patch_size, training=True)
    val_transforms = get_transforms(patch_size=patch_size, training=False)

    train_ds = MonaiDataset(data=train_data, transform=train_transforms)
    val_ds = MonaiDataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # hiter test
    ds_dir = r"C:\Users\erikv\OneDrive\Desktop\AMS izziv\data\nnUNet_raw\Dataset001_ImageCAS"
    if os.path.isdir(ds_dir):
        tl, vl = get_dataloaders(ds_dir, batch_size=1)
        batch = next(iter(tl))
        print("Image shape:", batch["image"].shape)
        print("Label shape:", batch["label"].shape)
    else:
        print("Set ds_dir to your nnU-Net v2 dataset path.")
