import os
import shutil
import json
import pandas as pd
import nibabel as nib

# KONFIGURACIJA

EXCEL_PATH = "data/archive/imageCAS_data_split.xlsx"
RAW_DIR = "data/imagecas_raw"
OUT_DIR = "data/nnUNet_raw/Dataset001_ImageCAS"

# FUNKCIJE

def ensure_dirs():
    os.makedirs(os.path.join(OUT_DIR, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "imagesTs"), exist_ok=True)


def convert_case_id(num):
    """Pretvori ID (npr. 1) v nnU-Net format: case_0001"""
    return f"case_{int(num):04d}"


def validate_pair(img_path, lbl_path):
    """Preveri, ali se slika in maska ujemata v dimenzijah."""
    img = nib.load(img_path)
    lbl = nib.load(lbl_path)
    if img.shape != lbl.shape:
        print(f"WARNING: Shape mismatch {img_path} vs {lbl_path}")


def copy_and_rename(img_src, lbl_src, img_dst, lbl_dst):
    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst)


# BRANJE SPLITA

def load_split_info():
    # Excel header je v vrstici 1
    df = pd.read_excel(EXCEL_PATH, header=1)

    # ORIGINALNA LOGIKA ZA CELOTEN DATASET
    #
    # df = df[["FileName", "Split-1"]]
    # df["FileName"] = df["FileName"].astype(str)
    # return df
    #
    

    # 
    # RAZVOJNA LOGIKA — UPORABI SAMO PRVIH 200 PRIMEROV
    # 
    df = df.iloc[:200].copy()          # vzemi prvih 200 primerov
    df["FileName"] = range(1, 201)     # prepiši ID-je → 1..200
    df["FileName"] = df["FileName"].astype(str)

    # ignoriramo originalne ID-je, obdržimo samo split
    df = df[["FileName", "Split-1"]]

    return df

# GLAVNI PROGRAM

def main():
    print("Loading split info...")
    df = load_split_info()

    print("Creating nnU-Net structure...")
    ensure_dirs()

    train_cases = []
    test_cases = []


    # RAZVOJNA LOGIKA — 80/20 SPLIT

    N = len(df)
    cutoff = int(0.8 * N)   # 80% train, 20% test

    for idx, row in df.iterrows():
        file_id = row["FileName"]

        img_path = os.path.join(RAW_DIR, f"{file_id}.img.nii.gz")
        lbl_path = os.path.join(RAW_DIR, f"{file_id}.label.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"Skipping missing pair: {file_id}")
            continue

        validate_pair(img_path, lbl_path)
        case_id = convert_case_id(file_id)

        if idx < cutoff:
            # TRAIN
            dst_img = os.path.join(OUT_DIR, "imagesTr", f"{case_id}_0000.nii.gz")
            dst_lbl = os.path.join(OUT_DIR, "labelsTr", f"{case_id}.nii.gz")
            copy_and_rename(img_path, lbl_path, dst_img, dst_lbl)
            train_cases.append(case_id)

        else:
            # TEST
            dst_img = os.path.join(OUT_DIR, "imagesTs", f"{case_id}_0000.nii.gz")
            shutil.copy(img_path, dst_img)
            test_cases.append(case_id)

    # USTVARI dataset.json

    dataset_json = {
        "name": "ImageCAS",
        "description": "Mini dataset for development",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "aneurysm"},
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in train_cases
        ],
        "test": [
            f"./imagesTs/{cid}_0000.nii.gz"
            for cid in test_cases
        ]
    }

    with open(os.path.join(OUT_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)


    # USTVARI splits_final.json (za nnU-Net v2)

    splits = [{
        "train": train_cases,
        "val": test_cases
    }]

    with open(os.path.join(OUT_DIR, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
