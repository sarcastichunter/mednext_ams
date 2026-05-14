import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize_3d, dilation, ball

from mednext.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from src.data.dataset_full import CTADatasetFullVolume


def extract_centerline(mask_tensor):
    mask = mask_tensor.cpu().numpy().astype(np.uint8)
    return skeletonize_3d(mask)


def dilate_mask(mask, radius=2):
    return dilation(mask, ball(radius))


def compute_topology_metrics(pred_cl, ref_cl, radius=2):
    pred_dil = dilate_mask(pred_cl, radius)
    ref_dil = dilate_mask(ref_cl, radius)

    matched_reference = ref_cl & pred_dil
    matched_extraction = pred_cl & ref_dil

    completeness = matched_reference.sum() / (ref_cl.sum() + 1e-8)
    correctness = matched_extraction.sum() / (pred_cl.sum() + 1e-8)
    quality = (completeness * correctness) / (
        completeness + correctness - completeness * correctness + 1e-8
    )

    return completeness, correctness, quality


def main():
    parser = argparse.ArgumentParser(description="Test MedNeXt — topology metrics")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pot do shranjenega checkpointa (.pt)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Pot do nnU-Net dataset direktorija")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Pot za shranjevanje metrik (JSON datoteka)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Kateri split evalvirati (privzeto: val)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Popravek: ključ je "model_state" (skladno s train_loop.py)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    all_C, all_K, all_Q = [], [], []

    print("Evalvacija...")
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            ref_mask_tensor = batch["mask"][0, 0].cpu()

            pred = model(img)
            pred = (pred > 0.5).float()[0, 0]

            pred_cl = extract_centerline(pred)
            ref_cl = extract_centerline(ref_mask_tensor)

            C, K, Q = compute_topology_metrics(pred_cl, ref_cl)
            all_C.append(float(C))
            all_K.append(float(K))
            all_Q.append(float(Q))

    results = {
        "completeness": float(np.mean(all_C)),
        "correctness": float(np.mean(all_K)),
        "quality": float(np.mean(all_Q)),
    }

    print(f"Completeness : {results['completeness']:.4f}")
    print(f"Correctness  : {results['correctness']:.4f}")
    print(f"Quality      : {results['quality']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrike shranjene: {args.output_path}")


if __name__ == "__main__":
    main()