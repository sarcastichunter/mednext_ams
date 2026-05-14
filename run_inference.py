import os
import shutil
import json
import argparse
import pandas as pd
import nibabel as nib


# -----------------------------------------------------------------------
# CLI ARGUMENTI
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretvori ImageCAS dataset v nnU-Net v2 format."
    )
    parser.add_argument("--excel_path", type=str,
                        default="data/archive/imageCAS_data_split.xlsx",
                        help="Pot do Excel datoteke s split informacijo")
    parser.add_argument("--raw_dir", type=str,
                        default="data/imagecas_raw",
                        help="Pot do surovih ImageCAS datotek (.img.nii.gz, .label.nii.gz)")
    parser.add_argument("--out_dir", type=str,
                        default="data/nnUNet_raw/Dataset001_ImageCAS",
                        help="Izhodna pot za nnU-Net strukturo")
    parser.add_argument("--split_col", type=str,
                        default="Split-1",
                        help="Ime stolpca v Excelu (Split-1, Split-2, ...)")
    return parser.parse_args()


# -----------------------------------------------------------------------
# POMOŽNE FUNKCIJE
# -----------------------------------------------------------------------

def ensure_dirs(out_dir: str):
    """Ustvari potrebne podmape za nnU-Net format."""
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def convert_case_id(num) -> str:
    """Pretvori številski ID v nnU-Net format: 1 → case_0001."""
    return f"case_{int(num):04d}"


def validate_pair(img_path: str, lbl_path: str):
    """
    Preveri, ali se slika in maska ujemata v dimenzijah.
    Ob neskladju sproži ValueError namesto samo opozorila.
    """
    img_shape = nib.load(img_path).shape
    lbl_shape = nib.load(lbl_path).shape
    if img_shape != lbl_shape:
        raise ValueError(
            f"Shape mismatch: {img_path} {img_shape} vs {lbl_path} {lbl_shape}"
        )


def load_split_info(excel_path: str, split_col: str) -> pd.DataFrame:
    """Prebere Excel in vrne DataFrame s stolpcema FileName in izbranim splitom."""
    df = pd.read_excel(excel_path, header=1)
    if split_col not in df.columns:
        raise ValueError(f"Stolpec '{split_col}' ni najden v Excelu. Dostopni stolpci: {list(df.columns)}")
    df = df[["FileName", split_col]].copy()
    df["FileName"] = df["FileName"].astype(str)
    return df


# -----------------------------------------------------------------------
# GLAVNI PROGRAM
# -----------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Berem split info iz: {args.excel_path} (stolpec: {args.split_col})")
    df = load_split_info(args.excel_path, args.split_col)

    print(f"Ustvarjam nnU-Net strukturo v: {args.out_dir}")
    ensure_dirs(args.out_dir)

    train_cases, val_cases, test_cases = [], [], []
    skipped = 0

    for idx, row in df.iterrows():
        # Indeks vrstice (+1) določa ime datoteke — FileName v Excelu se ne ujema z imenom datoteke
        file_id = str(idx + 1)

        img_path = os.path.join(args.raw_dir, f"{file_id}.img.nii.gz")
        lbl_path = os.path.join(args.raw_dir, f"{file_id}.label.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"  [SKIP] Manjkajoč par: {file_id}")
            skipped += 1
            continue

        try:
            validate_pair(img_path, lbl_path)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            skipped += 1
            continue

        case_id = convert_case_id(file_id)
        split = str(row[args.split_col]).strip()

        if split in ("Training", "Val"):
            dst_img = os.path.join(args.out_dir, "imagesTr", f"{case_id}_0000.nii.gz")
            dst_lbl = os.path.join(args.out_dir, "labelsTr", f"{case_id}.nii.gz")
            shutil.copy(img_path, dst_img)
            shutil.copy(lbl_path, dst_lbl)
            (train_cases if split == "Training" else val_cases).append(case_id)

        elif split == "Testing":
            dst_img = os.path.join(args.out_dir, "imagesTs", f"{case_id}_0000.nii.gz")
            shutil.copy(img_path, dst_img)
            test_cases.append(case_id)

        else:
            print(f"  [WARN] Neznan split label '{split}' za case {file_id}, preskakujem.")
            skipped += 1

    # --- dataset.json ---
    dataset_json = {
        "name": "ImageCAS",
        "description": "CTA dataset converted to nnU-Net format",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "coronary_artery"},  # popravek: bil "aneurysm"
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
    print(f"Shranjeno: {dataset_json_path}")

    # --- splits_final.json ---
    splits = [{"train": train_cases, "val": val_cases}]

    splits_path = os.path.join(args.out_dir, "splits_final.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Shranjeno: {splits_path}")

    print("\n--- Povzetek ---")
    print(f"  Training:  {len(train_cases)}")
    print(f"  Val:       {len(val_cases)}")
    print(f"  Testing:   {len(test_cases)}")
    print(f"  Preskočeni: {skipped}")
    print("Konverzija končana.")


if __name__ == "__main__":
    main()