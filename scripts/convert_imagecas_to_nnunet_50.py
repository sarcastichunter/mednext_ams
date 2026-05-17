"""
testna skripta za testiranje, naključnih 50 primerov in delitev 80/20

Uporaba:
    python3 scripts/convert_imagecas_to_nnunet_50.py \
        --raw_dir  /media/FastDataMama/izziv \
        --out_dir  data/nnUNet_raw/Dataset002_ImageCAS50 \
        --n_cases  50 \
        --seed     42
"""

import argparse
import json
import os
import random
import shutil

import nibabel as nib


# CLI ARGUMENTI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pripravi manjši testni dataset iz N naključnih ImageCAS primerov."
    )
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Pot do surovih ImageCAS datotek (.img.nii.gz, .label.nii.gz)")
    parser.add_argument("--out_dir", type=str,
                        default="data/nnUNet_raw/Dataset002_ImageCAS50",
                        help="Izhodna pot za nnU-Net strukturo")
    parser.add_argument("--n_cases", type=int, default=50,
                        help="Število primerov za izbiro (privzeto: 50)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Delež train primerov (privzeto: 0.8 → 80% train, 20% val)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed za naključno izbiro (privzeto: 42, za ponovljivost)")
    return parser.parse_args()


# POMOŽNE FUNKCIJE

def ensure_dirs(out_dir: str):
    """Ustvari potrebne podmape za nnU-Net format."""
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def convert_case_id(num) -> str:
    """Pretvori številski ID v nnU-Net format: 1 → case_0001."""
    return f"case_{int(num):04d}"


def validate_pair(img_path: str, lbl_path: str) -> bool:
    """
    Preveri ali se slika in maska ujemata v dimenzijah.
    Vrne True če je par ustrezen, False če ne.
    """
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
    
    available = []
 
    # Zberemo vse mape — tako raw_dir kot vse podmape
    search_dirs = [raw_dir]
    for entry in sorted(os.listdir(raw_dir)):
        full_path = os.path.join(raw_dir, entry)
        if os.path.isdir(full_path):
            search_dirs.append(full_path)
 
    print(f"  Iščem v mapah: {[os.path.basename(d) for d in search_dirs]}")
 
    for search_dir in search_dirs:
        try:
            files = os.listdir(search_dir)
        except PermissionError:
            continue
 
        img_files = [f for f in files if f.endswith(".img.nii.gz")]
 
        for img_file in img_files:
            file_id = img_file.replace(".img.nii.gz", "")
            lbl_file = f"{file_id}.label.nii.gz"
 
            img_path = os.path.join(search_dir, img_file)
            lbl_path = os.path.join(search_dir, lbl_file)
 
            if os.path.exists(lbl_path):
                available.append((file_id, img_path, lbl_path))
 
    # Sortiraj po file_id
    available.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return available

# GLAVNI PROGRAM

def main():
    args = parse_args()

    # Nastavi seed za ponovljivost
    random.seed(args.seed)

    print(f"Iščem veljavne pare v: {args.raw_dir}")
    available = find_available_cases(args.raw_dir)
    print(f"Najdenih {len(available)} veljavnih parov.")

    if len(available) < args.n_cases:
        print(f"  [WARN] Zahtevanih {args.n_cases} primerov, na voljo samo {len(available)}.")
        print(f"  Nadaljujem z vsemi {len(available)} primeri.")
        args.n_cases = len(available)

    # Naključna izbira n_cases primerov
    selected = random.sample(available, args.n_cases)
    selected.sort(key=lambda x: int(x) if x.isdigit() else x)

    # Razdelitev 80% train, 20% val
    n_train = int(args.n_cases * args.train_ratio)
    train_ids = selected[:n_train]
    val_ids   = selected[n_train:]

    print(f"\nIzbrano {args.n_cases} primerov (seed={args.seed}):")
    print(f"  Train: {len(train_ids)} primerov")
    print(f"  Val:   {len(val_ids)} primerov")

    print(f"\nUstvarjam nnU-Net strukturo v: {args.out_dir}")
    ensure_dirs(args.out_dir)

    train_cases, val_cases = [], []
    skipped = 0

    all_splits = [("Training", train_ids, train_cases), ("Val", val_ids, val_cases)]

    for split_name, ids, cases_list in all_splits:
        print(f"\nKopiram {split_name} primere...")

        for file_id in ids:
            img_path = os.path.join(args.raw_dir, f"{file_id}.img.nii.gz")
            lbl_path = os.path.join(args.raw_dir, f"{file_id}.label.nii.gz")

            if not validate_pair(img_path, lbl_path):
                skipped += 1
                continue

            case_id = convert_case_id(file_id)
            dst_img = os.path.join(args.out_dir, "imagesTr", f"{case_id}_0000.nii.gz")
            dst_lbl = os.path.join(args.out_dir, "labelsTr", f"{case_id}.nii.gz")

            shutil.copy(img_path, dst_img)
            shutil.copy(lbl_path, dst_lbl)
            cases_list.append(case_id)

            print(f"  [{split_name}] {file_id} → {case_id}")

    # --- dataset.json ---
    dataset_json = {
        "name": "ImageCAS_50",
        "description": f"Testni dataset: {args.n_cases} naključnih primerov, seed={args.seed}",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "coronary_artery"},
        "numTraining": len(train_cases) + len(val_cases),
        "numTest": 0,
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in train_cases + val_cases
        ],
        "test": [],
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
    print(f"  Preskočeni: {skipped}")
    print(f"  Seed:       {args.seed}")
    print(f"\nShranjeno:")
    print(f"  {dataset_json_path}")
    print(f"  {splits_path}")
    print("\nKonverzija končana.")
    print(f"\nZa trening uporabite:")
    print(f"  python3 run_train.py --data_dir {args.out_dir} --split_id 0")


if __name__ == "__main__":
    main()