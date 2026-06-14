"""
eval_nnunet.py — Evalvacija napovedi modela (nnU-Net ali MedNeXt).

Prebere napovedi (.nii.gz) in ground truth maske iz surovih podatkov
ter izračuna Dice, HD95, Completeness, Correctness in Quality.

Uporaba:
    python3 scripts/eval_nnunet.py \
        --pred_path  /workspace/predictions/nnunet_val \
        --raw_dir    /data/data \
        --output_path /workspace/results/nnunet_metrics.json
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from skimage.morphology import ball, dilation, skeletonize


# -----------------------------------------------------------------------
# TOPOLOŠKE METRIKE (Heipke 1998)
# -----------------------------------------------------------------------

def extract_centerline(mask_np: np.ndarray) -> np.ndarray:
    return skeletonize(mask_np.astype(np.uint8))


def dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
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
    return float(completeness), float(correctness), float(quality)


# -----------------------------------------------------------------------
# POMOŽNE FUNKCIJE
# -----------------------------------------------------------------------

def load_binary_mask(path: str) -> np.ndarray:
    """Naloži .nii.gz masko in jo binarizira (vrednosti > 0 → 1)."""
    arr = nib.load(path).get_fdata().astype(np.float32)
    return (arr > 0).astype(np.uint8)


def find_label_in_raw(file_id: str, raw_dir: str) -> str:
    """
    Poišče ground truth masko v surovih podatkih.
    Preišče vse podmape (1-200, 201-400, ...).
    Enako kot vizualizacija.py.
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


def build_pairs(pred_path: str, raw_dir: str):
    """
    Poišče pare (napoved, ground truth) kjer:
    - napovedi so case_XXXX.nii.gz v pred_path
    - ground truth so {file_id}.label.nii.gz v raw_dir in podmapah
    """
    pairs = []
    missing = []

    pred_files = sorted([f for f in os.listdir(pred_path) if f.endswith(".nii.gz")])

    for fname in pred_files:
        pred_file = os.path.join(pred_path, fname)

        # Pretvori case_0001.nii.gz → file_id = "1"
        case_id  = fname.replace(".nii.gz", "")
        file_id  = str(int(case_id.replace("case_", "")))

        lbl_path = find_label_in_raw(file_id, raw_dir)

        if lbl_path is None:
            print(f"  [SKIP] Ground truth ni najden za {case_id} (file_id={file_id})")
            missing.append(case_id)
            continue

        pairs.append((pred_file, lbl_path, case_id))

    if missing:
        print(f"  [WARN] {len(missing)} primerov brez ground truth.")

    return pairs


def to_monai_tensor(mask_np: np.ndarray) -> torch.Tensor:
    """numpy (D,H,W) → torch (1, 1, D, H, W) za MONAI metrike."""
    return torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)


# -----------------------------------------------------------------------
# GLAVNI PROGRAM
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evalvacija napovedi modela z ground truth iz surovih podatkov."
    )
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Mapa z napovedmi (case_XXXX.nii.gz)")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Mapa s surovi podatki ImageCAS (vsebuje podmape 1-200, 201-400 ...)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Pot za shranjevanje metrik (JSON)")
    parser.add_argument("--model", type=str, default="nnU-Net",
                        help="Ime modela za JSON (privzeto: nnU-Net)")
    args = parser.parse_args()

    print(f"Iskanje parov napoved / ground truth...")
    print(f"  Napovedi: {args.pred_path}")
    print(f"  Raw dir:  {args.raw_dir}")

    pairs = build_pairs(args.pred_path, args.raw_dir)
    print(f"Najdenih {len(pairs)} parov.\n")

    if not pairs:
        print("[ERROR] Nobenih parov — preverite poti.")
        return

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    all_C, all_K, all_Q = [], [], []
    n_hd95_skipped = 0

    for i, (pred_file, label_file, case_id) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {case_id}")

        pred_np  = load_binary_mask(pred_file)
        label_np = load_binary_mask(label_file)

        pred_t  = to_monai_tensor(pred_np)
        label_t = to_monai_tensor(label_np)

        # Dice
        dice_metric(y_pred=pred_t, y=label_t)

        # HD95
        if pred_np.sum() > 0 and label_np.sum() > 0:
            hd95_metric(y_pred=pred_t, y=label_t)
        else:
            n_hd95_skipped += 1

        # Topološke metrike
        pred_cl  = extract_centerline(pred_np)
        label_cl = extract_centerline(label_np)
        C, K, Q  = compute_topology_metrics(pred_cl, label_cl)

        all_C.append(C)
        all_K.append(K)
        all_Q.append(Q)

    # Zberi rezultate
    dice_scores = dice_metric.aggregate().numpy().flatten()
    hd95_scores = hd95_metric.aggregate().numpy().flatten()
    hd95_valid  = hd95_scores[~np.isnan(hd95_scores)]

    results = {
        "model":        args.model,
        "n_evaluated":  len(pairs),
        "dice_mean":    float(np.mean(dice_scores)),
        "dice_std":     float(np.std(dice_scores)),
        "hd95_mean":    float(np.mean(hd95_valid)) if len(hd95_valid) > 0 else None,
        "hd95_std":     float(np.std(hd95_valid))  if len(hd95_valid) > 0 else None,
        "hd95_skipped": n_hd95_skipped,
        "completeness": float(np.mean(all_C)),
        "correctness":  float(np.mean(all_K)),
        "quality":      float(np.mean(all_Q)),
    }

    print("\n--- Rezultati ---")
    print(f"Model         : {results['model']}")
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