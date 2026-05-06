import os
import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "data/nnUNet_raw/Dataset001_ImageCAS"

def load_case(case_id):
    img_path = os.path.join(DATASET_DIR, "imagesTr", f"{case_id}_0000.nii.gz")
    lbl_path = os.path.join(DATASET_DIR, "labelsTr", f"{case_id}.nii.gz")

    img = nib.load(img_path).get_fdata()
    lbl = nib.load(lbl_path).get_fdata()

    return img, lbl

def show_slice(img, lbl, slice_idx=None):
    if slice_idx is None:
        slice_idx = img.shape[2] // 2  # sredinski slice

    ct_slice = img[:, :, slice_idx]
    mask_slice = lbl[:, :, slice_idx]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(ct_slice, cmap="gray")
    axs[0].set_title("CT")

    axs[1].imshow(mask_slice, cmap="gray")
    axs[1].set_title("Mask")

    axs[2].imshow(ct_slice, cmap="gray")
    axs[2].imshow(mask_slice, cmap="Reds", alpha=0.4)
    axs[2].set_title("Overlay")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    # naloži seznam primerov
    cases = sorted([f.replace("_0000.nii.gz", "") 
                    for f in os.listdir(os.path.join(DATASET_DIR, "imagesTr"))])

    # izberi naključni primer
    case_id = random.choice(cases)
    print(f"Visualizing: {case_id}")

    img, lbl = load_case(case_id)

    print("Image shape:", img.shape)
    print("Label shape:", lbl.shape)
    print("Unique label values:", np.unique(lbl))

    show_slice(img, lbl)

if __name__ == "__main__":
    main()
