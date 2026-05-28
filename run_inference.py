import argparse
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


# -----------------------------------------------------------------------
# CLI ARGUMENTI
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MedNeXt inferenca: segmentacija koronarnih arterij na novih CTA slikah."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Mapa z vhodnimi slikami (*_0000.nii.gz ali *.nii.gz)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pot do naučenega checkpointa (.pt)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Mapa za shranjevanje napovedanih mask (.nii.gz)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("D", "H", "W"),
                        help="Velikost patcha (privzeto: 128 128 128)")
    parser.add_argument("--sw_overlap", type=float, default=0.5,
                        help="Prekrivanje med patchi (0.0-1.0, privzeto: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prag za binarizacijo napovedi (privzeto: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda ali cpu")
    return parser.parse_args()


# -----------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------

def preprocess(img_np: np.ndarray) -> torch.Tensor:
    """Enak preprocessing kot med treningom — HU okno [-1000, 1000] → [0.0, 1.0]."""
    img = img_np.astype(np.float32)
    img = np.clip(img, -1000, 1000)
    img = (img - (-1000)) / (1000 - (-1000))
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor


# -----------------------------------------------------------------------
# NALAGANJE SLIK
# -----------------------------------------------------------------------

def find_input_files(input_path: str):
    """Poišče vse .nii.gz datoteke v vhodni mapi."""
    files = []
    for fname in sorted(os.listdir(input_path)):
        if not fname.endswith(".nii.gz"):
            continue
        fpath = os.path.join(input_path, fname)
        if fname.endswith("_0000.nii.gz"):
            case_name = fname.replace("_0000.nii.gz", "")
        else:
            case_name = fname.replace(".nii.gz", "")
        files.append((case_name, fpath))
    return files


# -----------------------------------------------------------------------
# INICIALIZACIJA MODELA
# -----------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Inicializira MedNeXt in naloži uteži iz checkpointa."""
    model = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=1,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        checkpoint_style=None,
        dim='3d',
        grn=False,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------
# INFERENCA NA ENI SLIKI
# -----------------------------------------------------------------------

def run_inference_single(model, img_tensor, patch_size, sw_overlap, threshold, device):
    """Sliding window inferenca na eni sliki."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=img_tensor,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )
        pred = torch.sigmoid(pred)
    pred_np = pred[0, 0].cpu().numpy()
    binary_mask = (pred_np > threshold).astype(np.uint8)
    return binary_mask


# -----------------------------------------------------------------------
# GLAVNI PROGRAM
# -----------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA ni na voljo, preklapljam na CPU.")

    os.makedirs(args.output_path, exist_ok=True)
    patch_size = tuple(args.patch_size)

    print(f"Nalaganje modela iz: {args.model_path}")
    model = load_model(args.model_path, device)

    input_files = find_input_files(args.input_path)
    if not input_files:
        print(f"  [ERROR] Nobene .nii.gz datoteke v: {args.input_path}")
        return

    print(f"Najdenih {len(input_files)} slik. Začenjam inferenco...\n")

    for i, (case_name, img_path) in enumerate(input_files, 1):
        print(f"  [{i}/{len(input_files)}] {case_name}")

        nii_obj = nib.load(img_path)
        img_np = nii_obj.get_fdata().astype(np.float32)

        img_tensor = preprocess(img_np)

        binary_mask = run_inference_single(
            model=model,
            img_tensor=img_tensor,
            patch_size=patch_size,
            sw_overlap=args.sw_overlap,
            threshold=args.threshold,
            device=device,
        )

        out_nii = nib.Nifti1Image(binary_mask, affine=nii_obj.affine, header=nii_obj.header)
        out_path = os.path.join(args.output_path, f"{case_name}.nii.gz")
        nib.save(out_nii, out_path)
        print(f"    -> Shranjeno: {out_path}")

    print(f"\nInferenca končana. Napovedi shranjene v: {args.output_path}")


if __name__ == "__main__":
    main()