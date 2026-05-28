"""
vizualizacija.py

Primerja ground truth maske z napovedanimi maskami modela MedNeXt.
Shrani slike v mapo results/vizualizacije/.

Uporaba:
    python3 vizualizacija.py \
        --cases "1,2,3" \
        --raw_dir    /data/data/1-200 \
        --pred_dir   /workspace/predictions/mednext_val \
        --output_dir /workspace/results/vizualizacije
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# CLI ARGUMENTI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vizualna primerjava ground truth in napovedi MedNeXt."
    )
    parser.add_argument("--cases", type=str, required=True,
                        help="ID-ji primerov ločeni z vejico (npr. '1,2,3')")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Mapa s surovima slikami (.img.nii.gz in .label.nii.gz)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Mapa z napovedanimi maskami iz run_inference.py")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/results/vizualizacije",
                        help="Mapa za shranjevanje slik (privzeto: /workspace/results/vizualizacije)")
    return parser.parse_args()


# POMOŽNE FUNKCIJE

def load_nii(path):
    """Naloži NIfTI datoteko in vrne numpy array."""
    return nib.load(path).get_fdata().astype(np.float32)


def normalize_display(img):
    """Normalizira CTA sliko za prikaz v [0, 1]."""
    img = np.clip(img, -1000, 1000)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def find_best_slice(mask):
    """Poišče aksijalno rezino z največ pozitivnimi voksli."""
    sums = mask.sum(axis=(0, 1))
    return int(np.argmax(sums))


def compute_dice(pred, gt):
    """Izračuna Dice score."""
    intersection = (pred * gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)


# VIZUALIZACIJA

def visualize_case(case_id, raw_dir, pred_dir, output_dir):
    """
    Za en primer prikaže 4 stolpce:
    CTA original | Ground Truth | Napoved MedNeXt | Analiza napak
    """
    # Poti do datotek
    img_path  = os.path.join(raw_dir,  f"{case_id}.img.nii.gz")
    lbl_path  = os.path.join(raw_dir,  f"{case_id}.label.nii.gz")
    pred_path = os.path.join(pred_dir, f"case_{int(case_id):04d}.nii.gz")

    # Preverite da datoteke obstajajo
    for path, name in [(img_path, "slika"), (lbl_path, "ground truth"), (pred_path, "napoved")]:
        if not os.path.exists(path):
            print(f"  [SKIP] Primer {case_id} — manjka {name}: {path}")
            return

    print(f"  Obdelujem primer {case_id}...")

    # Nalaganje
    img  = load_nii(img_path)
    gt   = (load_nii(lbl_path)  > 0).astype(np.uint8)
    pred = (load_nii(pred_path) > 0).astype(np.uint8)

    # Dice score
    dice = compute_dice(pred, gt)

    # Najboljša rezina glede na ground truth
    z = find_best_slice(gt)

    img_s  = normalize_display(img[:, :, z])
    gt_s   = gt[:, :, z]
    pred_s = pred[:, :, z]

    # Napake
    tp = (pred_s == 1) & (gt_s == 1)  # pravilno
    fp = (pred_s == 1) & (gt_s == 0)  # False Positive
    fn = (pred_s == 0) & (gt_s == 1)  # False Negative

    # PRIKAZ — 4 stolpci

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Primer {case_id}  |  Rezina z={z}  |  Dice = {dice:.4f}",
        fontsize=13, fontweight='bold'
    )

    # 1. Originalna CTA slika
    axes[0].imshow(img_s.T, cmap='gray', origin='lower')
    axes[0].set_title('CTA original', fontsize=11)
    axes[0].axis('off')

    # 2. Ground Truth (zelena)
    axes[1].imshow(img_s.T, cmap='gray', origin='lower')
    ov_gt = np.zeros((*img_s.shape, 4))
    ov_gt[gt_s > 0] = [0.0, 1.0, 0.0, 0.5]
    axes[1].imshow(ov_gt.transpose(1, 0, 2), origin='lower')
    axes[1].set_title('Ground Truth', fontsize=11)
    axes[1].axis('off')

    # 3. Napoved MedNeXt (cian)
    axes[2].imshow(img_s.T, cmap='gray', origin='lower')
    ov_pred = np.zeros((*img_s.shape, 4))
    ov_pred[pred_s > 0] = [0.0, 0.8, 1.0, 0.5]
    axes[2].imshow(ov_pred.transpose(1, 0, 2), origin='lower')
    axes[2].set_title('Napoved MedNeXt', fontsize=11)
    axes[2].axis('off')

    # 4. Analiza napak
    axes[3].imshow(img_s.T, cmap='gray', origin='lower')
    ov_err = np.zeros((*img_s.shape, 4))
    ov_err[tp] = [0.0, 1.0, 0.0, 0.6]  # zelena = TP
    ov_err[fp] = [1.0, 0.0, 0.0, 0.6]  # rdeča  = FP
    ov_err[fn] = [0.0, 0.0, 1.0, 0.6]  # modra  = FN
    axes[3].imshow(ov_err.transpose(1, 0, 2), origin='lower')
    axes[3].set_title('Analiza napak', fontsize=11)
    axes[3].axis('off')

    # Legenda
    tp_p = mpatches.Patch(color='green', label='True Positive')
    fp_p = mpatches.Patch(color='red',   label='False Positive')
    fn_p = mpatches.Patch(color='blue',  label='False Negative')
    axes[3].legend(handles=[tp_p, fp_p, fn_p], loc='lower right', fontsize=8)

    plt.tight_layout()

    # Shrani
    save_path = os.path.join(output_dir, f"primer_{case_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Dice: {dice:.4f} | Shranjeno: {save_path}")



# GLAVNI PROGRAM


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Razčleni ID-je primerov
    case_ids = [c.strip() for c in args.cases.split(",")]

    print(f"Vizualizacija {len(case_ids)} primerov...")
    print(f"  Raw dir:  {args.raw_dir}")
    print(f"  Pred dir: {args.pred_dir}")
    print(f"  Output:   {args.output_dir}\n")

    for case_id in case_ids:
        visualize_case(case_id, args.raw_dir, args.pred_dir, args.output_dir)

    print(f"\nKončano. Slike shranjene v: {args.output_dir}")


if __name__ == "__main__":
    main()