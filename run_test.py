import argparse
import json
import os
import torch
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize, dilation, ball
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


# -----------------------------------------------------------------------
# TOPOLOŠKE METRIKE (Heipke 1998)
# -----------------------------------------------------------------------

def extract_centerline(mask_np):
    return skeletonize(mask_np.astype(np.uint8))


def dilate_mask(mask, radius=2):
    return dilation(mask, ball(radius))


def compute_topology_metrics(pred_cl, ref_cl, radius=2):
    pred_dil = dilate_mask(pred_cl, radius)
    ref_dil  = dilate_mask(ref_cl,  radius)

    matched_reference  = ref_cl  & pred_dil
    matched_extraction = pred_cl & ref_dil

    completeness = matched_reference.sum()  / (ref_cl.sum()  + 1e-8)
    correctness  = matched_extraction.sum() / (pred_cl.sum() + 1e-8)
    quality = (completeness * correctness) / (
        completeness + correctness - completeness * correctness + 1e-8
    )
    return completeness, correctness, quality


# -----------------------------------------------------------------------
# POMOŽNE FUNKCIJE
# -----------------------------------------------------------------------

def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def preprocess(img_np: np.ndarray) -> torch.Tensor:
    """HU okno [-1000, 1000] → [0.0, 1.0] — enako kot med treningom."""
    img = np.clip(img_np, -1000, 1000)
    img = (img - (-1000)) / (1000 - (-1000))
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)


def find_label_in_raw(file_id: str, raw_dir: str) -> str:
    """
    Poišče ground truth masko v surovih podatkih.
    Preišče vse podmape (1-200, 201-400, ...).
    """
    search_dirs = [raw_dir]
    for entry in sorted(os.listdir(raw_dir)):
        full_path = os.path.join(raw_dir, entry)
        if os.path.isdir(full_path):
            search_dirs.append(full_path)

    for search_dir in search_dirs:
        lbl_path = os.path.join(search_dir, f"{file_id}.label.nii.gz")
        if os.path.exists(lbl_path):
            return lbl_path
    return None


def find_image_in_raw(file_id: str, raw_dir: str) -> str:
    """Poišče originalno CTA sliko v surovih podatkih."""
    search_dirs = [raw_dir]
    for entry in sorted(os.listdir(raw_dir)):
        full_path = os.path.join(raw_dir, entry)
        if os.path.isdir(full_path):
            search_dirs.append(full_path)

    for search_dir in search_dirs:
        img_path = os.path.join(search_dir, f"{file_id}.img.nii.gz")
        if os.path.exists(img_path):
            return img_path
    return None


def to_monai_tensor(mask_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)


# -----------------------------------------------------------------------
# GLAVNI PROGRAM
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test MedNeXt — Dice, HD95 in topološke metrike")
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Pot do shranjenega checkpointa (.pt)")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Pot do nnU-Net dataset direktorija (vsebuje imagesTs/)")
    parser.add_argument("--raw_dir",     type=str, required=True,
                        help="Pot do surovih ImageCAS podatkov (vsebuje podmape 1-200, 201-400 ...)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Pot za shranjevanje metrik (JSON datoteka)")
    parser.add_argument("--patch_size",  type=int, nargs=3, default=[128, 128, 128],
                        metavar=("D", "H", "W"),
                        help="Velikost patcha za sliding window (privzeto: 128 128 128)")
    parser.add_argument("--sw_overlap",  type=float, default=0.5,
                        help="Prekrivanje pri sliding window (privzeto: 0.5)")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="cuda ali cpu")
    args = parser.parse_args()

    device     = args.device if torch.cuda.is_available() else "cpu"
    patch_size = tuple(args.patch_size)

    # Poišči vse testne primere v imagesTs/
    images_ts_dir = os.path.join(args.data_path, "imagesTs")
    test_files    = sorted([f for f in os.listdir(images_ts_dir) if f.endswith("_0000.nii.gz")])

    if not test_files:
        print(f"[ERROR] Nobenih datotek v {images_ts_dir}")
        return

    print(f"Najdenih {len(test_files)} testnih primerov.")

    # Naloži model
    print(f"Nalaganje modela iz: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

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
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    all_C, all_K, all_Q = [], [], []
    n_hd95_skipped = 0
    skipped = 0

    print(f"Evalvacija s sliding window (patch={patch_size}, overlap={args.sw_overlap})...\n")

    with torch.no_grad():
        for i, fname in enumerate(test_files, 1):
            case_id = fname.replace("_0000.nii.gz", "")
            file_id = str(int(case_id.replace("case_", "")))

            print(f"  [{i}/{len(test_files)}] {case_id}")

            # Poišči ground truth masko v surovih podatkih
            lbl_path = find_label_in_raw(file_id, args.raw_dir)
            if lbl_path is None:
                print(f"    [SKIP] Ground truth ni najden za file_id={file_id}")
                skipped += 1
                continue

            # Naloži sliko iz imagesTs
            img_path = os.path.join(images_ts_dir, fname)
            img_np   = load_nii(img_path)
            img_t    = preprocess(img_np).to(device)

            # Naloži ground truth masko
            lbl_np = (load_nii(lbl_path) > 0).astype(np.uint8)

            # Sliding window inferenca
            pred_logits = sliding_window_inference(
                inputs=img_t,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=args.sw_overlap,
                mode="gaussian",
            )
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
            pred_np     = pred_binary[0, 0].cpu().numpy()

            # MONAI metrike
            pred_t  = to_monai_tensor(pred_np)
            label_t = to_monai_tensor(lbl_np)

            dice_metric(y_pred=pred_t, y=label_t)

            if pred_np.sum() > 0 and lbl_np.sum() > 0:
                hd95_metric(y_pred=pred_t, y=label_t)
            else:
                n_hd95_skipped += 1

            # Topološke metrike
            pred_cl  = extract_centerline(pred_np)
            ref_cl   = extract_centerline(lbl_np)
            C, K, Q  = compute_topology_metrics(pred_cl, ref_cl)

            all_C.append(float(C))
            all_K.append(float(K))
            all_Q.append(float(Q))

    # Zberi rezultate
    dice_scores = dice_metric.aggregate().cpu().numpy().flatten()
    hd95_scores = hd95_metric.aggregate().cpu().numpy().flatten()
    hd95_valid  = hd95_scores[~np.isnan(hd95_scores)]

    results = {
        "dice_mean":    float(np.mean(dice_scores)),
        "dice_std":     float(np.std(dice_scores)),
        "hd95_mean":    float(np.mean(hd95_valid)) if len(hd95_valid) > 0 else None,
        "hd95_std":     float(np.std(hd95_valid))  if len(hd95_valid) > 0 else None,
        "completeness": float(np.mean(all_C)),
        "correctness":  float(np.mean(all_K)),
        "quality":      float(np.mean(all_Q)),
        "n_evaluated":  len(dice_scores),
        "n_hd95_valid": int(len(hd95_valid)),
        "n_skipped":    skipped,
    }

    print("\n--- Rezultati ---")
    print(f"Dice          : {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    if results["hd95_mean"] is not None:
        print(f"HD95          : {results['hd95_mean']:.4f} ± {results['hd95_std']:.4f}")
    else:
        print("HD95          : N/A")
    print(f"Completeness  : {results['completeness']:.4f}")
    print(f"Correctness   : {results['correctness']:.4f}")
    print(f"Quality       : {results['quality']:.4f}")
    print(f"Evalviranih   : {results['n_evaluated']} primerov")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nMetrike shranjene: {args.output_path}")


if __name__ == "__main__":
    main()