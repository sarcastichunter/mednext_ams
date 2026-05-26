"""
convert_imagecas_to_nnunet_n_samples.py

Pripravi dataset iz N naključnih primerov ImageCAS z nastavljivo razdelitvijo
na train, val in test množico. 

Uporaba:
    python3 scripts/convert_imagecas_to_nnunet_n_samples.py \
        --raw_dir     /data/data \
        --out_dir     /workspace/data/nnUNet_raw/Dataset003_ImageCAS \
        --n_cases     200 \
        --train_ratio 0.7 \
        --val_ratio   0.2 \
        --test_ratio  0.1 \
        --seed        42
"""

import argparse
import json
import os
import random
import shutil

import nibabel as nib


# -----------------------------------------------------------------------
# CLI ARGUMENTI
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pripravi dataset iz N naključnih ImageCAS primerov z nastavljivo razdelitvijo."
    )
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Pot do surovih ImageCAS datotek (.img.nii.gz, .label.nii.gz)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Izhodna pot za nnU-Net strukturo")
    parser.add_argument("--n_cases", type=int, default=200,
                        help="Število primerov za izbiro (privzeto: 200)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Delež train primerov (privzeto: 0.7 → 70%%)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Delež val primerov (privzeto: 0.2 → 20%%)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Delež test primerov (privzeto: 0.1 → 10%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed za naključno izbiro (privzeto: 42)")
    return parser.parse_args()


# VALIDACIJA ARGUMENTOV

def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    """Preveri da vsota razmerij ne preseže 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Vsota train_ratio + val_ratio + test_ratio mora biti 1.0, "
            f"dobljeno: {train_ratio} + {val_ratio} + {test_ratio} = {total:.4f}"
        )


# POMOŽNE FUNKCIJE

def ensure_dirs(out_dir: str):
    """Ustvari potrebne podmape za nnU-Net format."""
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def convert_case_id(num) -> str:
    """Pretvori številski ID v nnU-Net format: 1 → case_0001."""
    return f"case_{int(num):04d}"


def validate_pair(img_path: str, lbl_path: str) -> bool:
    """Preveri ali se slika in maska ujemata v dimenzijah."""
    try:
        img_shape = nib.load(img_path).shape
        lbl_shape = nib.load(lbl_path).shape
        if img_shape != lbl_shape:
            print(f"  [SKIP] Shape mismatch: {img_path} {img_shape} vs {lbl_path} {lbl_shape}")
            return False
        return True
    except Exception as e:
        print(f"  [SKIP] Napaka pri branju: {e}")
        return False


def find_available_cases(raw_dir: str):
    """
    Poišče vse veljavne pare (slika + maska) v raw_dir in vseh podmapah.
    Podpira strukturo: raw_dir/1-200/, raw_dir/201-400/, itd.
    Vrne seznam tuplev (file_id, img_path, lbl_path).
    """
    available = []

    search_dirs = [raw_dir]
    for entry in sorted(os.listdir(raw_dir)):
        full_path = os.path.join(raw_dir, entry)
        if os.path.isdir(full_path):
            search_dirs.append(full_path)

    print(f"  Iščem v mapah: {[os.path.basename(d) or d for d in search_dirs]}")

    for search_dir in search_dirs:
        try:
            files = os.listdir(search_dir)
        except PermissionError:
            continue

        for fname in sorted(files):
            if not fname.endswith(".img.nii.gz"):
                continue

            file_id  = fname.replace(".img.nii.gz", "")
            img_path = os.path.join(search_dir, fname)
            lbl_path = os.path.join(search_dir, f"{file_id}.label.nii.gz")

            if os.path.exists(lbl_path):
                available.append((file_id, img_path, lbl_path))

    available.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return available


# KOPIRANJE PRIMEROV

def copy_cases(ids, split_name, out_dir, dest_img_dir, dest_lbl_dir, has_labels=True):
    """Kopira primere v ustrezno mapo in vrne seznam case_id-jev."""
    cases = []
    skipped = 0

    print(f"\nKopiram {split_name} primere ({len(ids)})...")

    for file_id, img_path, lbl_path in ids:

        if has_labels and not validate_pair(img_path, lbl_path):
            skipped += 1
            continue

        case_id = convert_case_id(file_id)
        dst_img = os.path.join(out_dir, dest_img_dir, f"{case_id}_0000.nii.gz")
        shutil.copy(img_path, dst_img)

        if has_labels:
            dst_lbl = os.path.join(out_dir, dest_lbl_dir, f"{case_id}.nii.gz")
            shutil.copy(lbl_path, dst_lbl)

        cases.append(case_id)
        print(f"  [{split_name}] {file_id} → {case_id}")

    return cases, skipped


# GLAVNI PROGRAM

def main():
    args = parse_args()

    # Validacija razmerij
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    # Seed za ponovljivost
    random.seed(args.seed)

    print(f"Iščem veljavne pare v: {args.raw_dir}")
    available = find_available_cases(args.raw_dir)
    print(f"Najdenih {len(available)} veljavnih parov.")

    # Prilagodi n_cases če ni dovolj podatkov
    if len(available) < args.n_cases:
        print(f"  [WARN] Zahtevanih {args.n_cases}, na voljo samo {len(available)}.")
        args.n_cases = len(available)

    # Naključna izbira
    selected = random.sample(available, args.n_cases)
    selected.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    # Razdelitev po razmerjih
    n_train = int(args.n_cases * args.train_ratio)
    n_val   = int(args.n_cases * args.val_ratio)
    n_test  = args.n_cases - n_train - n_val  # preostanek gre v test

    train_ids = selected[:n_train]
    val_ids   = selected[n_train:n_train + n_val]
    test_ids  = selected[n_train + n_val:]

    print(f"\nRazdelitev {args.n_cases} primerov (seed={args.seed}):")
    print(f"  Train: {len(train_ids)} ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_ids)}   ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_ids)}  ({args.test_ratio*100:.0f}%)")

    print(f"\nUstvarjam nnU-Net strukturo v: {args.out_dir}")
    ensure_dirs(args.out_dir)

    # Kopiranje
    train_cases, skip_tr = copy_cases(train_ids, "Training", args.out_dir, "imagesTr", "labelsTr")
    val_cases,   skip_v  = copy_cases(val_ids,   "Val",      args.out_dir, "imagesTr", "labelsTr")
    test_cases,  skip_ts = copy_cases(test_ids,  "Testing",  args.out_dir, "imagesTs", None, has_labels=False)

    total_skipped = skip_tr + skip_v + skip_ts

    # --- dataset.json ---
    dataset_json = {
        "name": f"ImageCAS_{args.n_cases}",
        "description": (
            f"{args.n_cases} naključnih primerov — "
            f"train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, "
            f"test={args.test_ratio:.0%}, seed={args.seed}"
        ),
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "coronary_artery"},
        "numTraining": len(train_cases) + len(val_cases),
        "numTest": len(test_cases),
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in train_cases + val_cases
        ],
        "test": [
            f"./imagesTs/{cid}_0000.nii.gz"
            for cid in test_cases
        ],
    }

    dataset_json_path = os.path.join(args.out_dir, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    # --- splits_final.json ---
    splits = [{"train": train_cases, "val": val_cases}]

    splits_path = os.path.join(args.out_dir, "splits_final.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n--- Povzetek ---")
    print(f"  Train:      {len(train_cases)} primerov")
    print(f"  Val:        {len(val_cases)} primerov")
    print(f"  Test:       {len(test_cases)} primerov  ← za run_inference.py")
    print(f"  Preskočeni: {total_skipped}")
    print(f"\nShranjeno:")
    print(f"  {dataset_json_path}")
    print(f"  {splits_path}")
    print(f"\nNaslednji koraki:")
    print(f"  Trening:    python3 run_train.py --data_dir {args.out_dir} --split_id 0")
    print(f"  Inferenca:  python3 run_inference.py --input_path {args.out_dir}/imagesTs ...")
    print(f"\nKonverzija končana.")


if __name__ == "__main__":
    main()