import os
import json
import torch
import nibabel as nib
from torch.utils.data import Dataset


class CTADatasetFullVolume(Dataset):
    """
    Full-volume dataset for CTA segmentation.
    Uses official nnU-Net split from splits_final.json.
    
    Expected structure:
    nnUNet_raw/
        Dataset001_ImageCAS/
            imagesTr/
            labelsTr/
            imagesTs/
            splits_final.json
    """

    def __init__(self, root, split="train"):
        """
        root: path to Dataset001_ImageCAS
        split: "train", "val", or "test"
        """
        self.root = root
        self.split = split

        if split in ["train", "val"]:
            self.image_dir = os.path.join(root, "imagesTr")
            self.label_dir = os.path.join(root, "labelsTr")

            # Load official nnU-Net split
            split_file = os.path.join(root, "splits_final.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Missing splits_final.json at: {split_file}")

            with open(split_file, "r") as f:
                splits = json.load(f)[0]  # nnU-Net stores list of splits

            if split == "train":
                self.cases = splits["train"]
            else:
                self.cases = splits["val"]

        elif split == "test":
            self.image_dir = os.path.join(root, "imagesTs")
            self.cases = sorted(os.listdir(self.image_dir))
            self.cases = [c.replace("_0000.nii.gz", "") for c in self.cases]

        else:
            raise ValueError("split must be train, val, or test")

    def __len__(self):
        return len(self.cases)

    def load_nii(self, path):
        nii = nib.load(path)
        arr = nii.get_fdata().astype("float32")
        return arr

    def __getitem__(self, idx):
        case_id = self.cases[idx]

        img_path = os.path.join(self.image_dir, f"{case_id}_0000.nii.gz")
        image = self.load_nii(img_path)

        # normalize / popravek, zamenjava mean/std z HU oknom
        image = np.clip(image, -1000, 1000)
        image = (image - (-1000)) / (1000 - (-1000))  # → [0.0, 1.0]
        image = torch.from_numpy(image).unsqueeze(0)
        
        if self.split == "test":
            return {"image": image, "case_id": case_id}

        # load mask
        mask_path = os.path.join(self.label_dir, f"{case_id}.nii.gz")
        mask = self.load_nii(mask_path)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "case_id": case_id
        }
