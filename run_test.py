import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize, dilation, ball
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from src.data.dataset_full import CTADatasetFullVolume


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
# GLAVNI PROGRAM
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test MedNeXt — Dice, HD95 in topološke metrike")
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Pot do shranjenega checkpointa (.pt)")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Pot do nnU-Net dataset direktorija")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Pot za shranjevanje metrik (JSON datoteka)")
    parser.add_argument("--split",       type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Kateri split evalvirati (privzeto: val)")
    parser.add_argument("--patch_size",  type=int, nargs=3, default=[128, 128, 128],
                        metavar=("D", "H", "W"),
                        help="Velikost patcha za sliding window (privzeto: 128 128 128)")
    parser.add_argument("--sw_overlap",  type=float, default=0.5,
                        help="Prekrivanje pri sliding window (privzeto: 0.5)")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="cuda ali cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    patch_size = tuple(args.patch_size)

    print(f"Nalaganje dataseta (split={args.split})...")
    dataset = CTADatasetFullVolume(root=args.data_path, split=args.split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    # MONAI metrike
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    all_C, all_K, all_Q = [], [], []
    n_hd95_skipped = 0

    print(f"Evalvacija s sliding window (patch={patch_size}, overlap={args.sw_overlap})...")
    with torch.no_grad():
        for batch in dataloader:
            img            = batch["image"].to(device)
            ref_mask_tensor = batch["mask"][0, 0].cpu().numpy()

            # Sliding window inferenca — reši OOM problem
            pred_logits = sliding_window_inference(
                inputs=img,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=args.sw_overlap,
                mode="gaussian",
            )

            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
            ref_binary  = batch["mask"].to(device)

            # Dice
            dice_metric(y_pred=pred_binary, y=ref_binary)

            # HD95
            if pred_binary.sum() > 0 and ref_binary.sum() > 0:
                hd95_metric(y_pred=pred_binary, y=ref_binary)
            else:
                n_hd95_skipped += 1

            # Topološke metrike na CPU
            pred_np = pred_binary[0, 0].cpu().numpy()
            pred_cl = extract_centerline(pred_np)
            ref_cl  = extract_centerline(ref_mask_tensor)

            C, K, Q = compute_topology_metrics(pred_cl, ref_cl)
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