# MedNeXt — Segmentacija koronarnih arterij (AMS Izziv 2025)

V dokumentu je predstavljena implementacija metode MedNeXt za avtomatsko segmentacijo koronarnih arterij na 3D CTA slikah, ki je bila izdelana v sklopu izziva pri predmetu Analiza medicinskih slik. Izvedli smo kvantitativno in kvalitativno primerjavo MedNeXt z nnU-Net, kot obveznim baseline modelom, na podatkovni zbirki ImageCAS.

## O metodi — MedNeXt

MedNeXt je popolnoma konvolucijska 3D arhitektura, ki temelji na modernizaciji standardnih U-Net blokov po vzoru ConvNeXt. Značilnost arhitekture Mednext je uporaba velikih jeder (large kernel) v kombinaciji s prilagodljivim ekspanzijskim razmerjem (exp_r) znotraj posameznih stopenj enkoder-dekoder strukture, kar modelu omogoča učinkovito sledenje dolgim, tankim in vijugastim strukturam — kar je problem pri segmentaciji koronarnih arterij.

V tej implementaciji je uporabljena 9-stopenjska MedNeXt arhitektura (`exp_r=[2,3,4,4,4,4,4,3,2]`, `n_channels=32`) z rezidualnimi povezavami tako v enkoderju kot dekoderju. Model je treniran z izgubo `BCEWithLogitsLoss` in optimizatorjem AdamW, z mešano natančnostjo (mixed precision) za hitrejše učenje. Zaradi velikosti CTA volumnov (do 512×512×500 vokslov) inferenca poteka s sliding window pristopom (patch 128×128×128, Gaussovo uteženo združevanje robov).

## Struktura repozitorija

```
.
├── Dockerfile
├── requirements.txt
├── README.md
├── run_train.py                          ← CLI treniranje MedNeXt
├── run_test.py                           ← CLI evalvacija na testni množici
├── run_inference.py                      ← CLI inferenca na novih slikah
├── scripts/
│   ├── convert_imagecas_to_nnunet.py     ← pretvorba ImageCAS → nnU-Net (uradni Excel split)
│   ├── convert_imagecas_to_nnunet_n_samples.py  ← pretvorba poljubnega N primerov
│   ├── eval_nnunet.py                    ← evalvacija napovedi glede na surove podatke
│   └── vizualizacija.py                  ← kvalitativna primerjava (CTA + GT + napoved)
├── src/
│   ├── data/
│   │   ├── dataset_full.py               ← full-volume dataloader (za run_test.py)
│   │   └── dataset_nnunet.py             ← patch dataloader z MONAI augmentacijami
│   └── trening/
│       └── train_loop.py                 ← Trainer razred (mixed precision, checkpointing)
└── run_all_folds.sh                      ← zagon treninga in evalvacije na vseh 4 foldih
```

Opomba: MedNeXt arhitektura (`nnunet_mednext`) se namesti direktno iz uradnega GitHub repozitorija znotraj Dockerfile-a in ni del tega repozitorija.

## Namestitev

### Zahteve
- Docker z NVIDIA Container Toolkit
- NVIDIA GPU (testirano na RTX 2080 Ti / RTX 4060 Ti)

### Gradnja Docker imagea

```bash
cd <repo>
docker build -t mednext_izziv .
```

Dockerfile namesti vse Python odvisnosti (`requirements.txt`), nnU-Net (`nnunetv2`) in MedNeXt arhitekturo (`nnunet_mednext` iz GitHub repozitorija MIC-DKFZ). Koda se ne kopira v image — med zagonom se zmapira z `-v`, kar omogoča urejanje kode brez ponovne gradnje image-a.

## Priprava podatkov

Podatki ImageCAS morajo biti v surovi obliki (`<id>.img.nii.gz`, `<id>.label.nii.gz`), lahko tudi razdeljeni v podmape (npr. `1-200/`, `201-400/` ...) — skripta jih sama poišče.

### Uradni Split-1 (za poročanje rezultatov)

```bash
docker run --gpus all --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 scripts/convert_imagecas_to_nnunet.py \
    --raw_dir    /data/data \
    --excel_path /data/imageCAS_data_split.xlsx \
    --out_dir    /workspace/data/nnUNet_raw/Dataset001_ImageCAS \
    --split_col  Split-1
```

Za ostale folde (4-kratna prečna validacija) zamenjaj `--split_col` z `Split-2`, `Split-3` ali `Split-4`.

### Manjši podvzorec (za hitro testiranje pipeline-a)

```bash
docker run --gpus all --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 scripts/convert_imagecas_to_nnunet_n_samples.py \
    --raw_dir     /data/data \
    --out_dir     /workspace/data/nnUNet_raw/Dataset002_ImageCAS200 \
    --n_cases     200 \
    --train_ratio 0.7 \
    --val_ratio   0.2 \
    --test_ratio  0.1 \
    --seed        42
```

## Treniranje

Trening je dolgotrajen proces (več ur do dni) — priporočen je zagon v `tmux` seji, da se ob prekinitvi SSH povezave ne izgubi napredek.

```bash
tmux new -s mednext_trening

docker run --gpus 'device=0' --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 run_train.py \
    --data_dir   /workspace/data/nnUNet_raw/Dataset001_ImageCAS \
    --epochs     120 \
    --batch_size 1 \
    --lr         1e-4 \
    --save_dir   /workspace/checkpoints/fold_0 \
    --split_id   0

# Odlepitev seje: Ctrl+B, nato D
```

`--split_id` določa kateri fold se uporabi za train/val razdelitev (0 = Split-1, 1 = Split-2, itd.). Checkpoint se shrani vsakič, ko se izboljša validacijska izguba.

### Vsi 4 foldi naenkrat

```bash
bash run_all_folds.sh \
    data/nnUNet_raw/Dataset001_ImageCAS \
    results/ \
    120 \
    cuda
```

---

## Testiranje (kvantitativna evalvacija)

`run_test.py` izvede sliding window inferenco na testni množici (`imagesTs/`) in primerja napovedi z ground truth maskami, ki jih poišče direktno v surovih podatkih (`--raw_dir`). Izračuna Dice, HD95 ter topološke metrike Completeness, Correctness in Quality (Heipke).

```bash
docker run --gpus 'device=0' --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 run_test.py \
    --model_path  /workspace/checkpoints/fold_0/epoch_<N>_loss_<L>.pt \
    --data_path   /workspace/data/nnUNet_raw/Dataset001_ImageCAS \
    --raw_dir     /data/data \
    --output_path /workspace/results/mednext_metrics.json
```

## Inferenca na novih slikah

`run_inference.py` naredi napoved segmentacijske maske na slikah brez ground truth (npr. nove paciente) in jih shrani kot `.nii.gz` z originalno geometrijo slike.

```bash
docker run --gpus 'device=0' --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 run_inference.py \
    --input_path  /workspace/data/nnUNet_raw/Dataset001_ImageCAS/imagesTs \
    --model_path  /workspace/checkpoints/fold_0/epoch_<N>_loss_<L>.pt \
    --output_path /workspace/predictions/mednext
```

## Primerjava z nnU-Net (baseline)

### 1. Preprocessing in trening nnU-Net

```bash
docker run --gpus 'device=1' --shm-size=32g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    -e nnUNet_raw=/workspace/data/nnUNet_raw \
    -e nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed \
    -e nnUNet_results=/workspace/data/nnUNet_results \
    mednext_izziv nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

Za pošteno primerjavo z nnU-Net moramo uporabiti enak train/val split kot pri MedNeXt — po preprocessingu kopirajte `splits_final.json`:

```bash
docker run --rm -v "$(pwd)":/workspace mednext_izziv \
    cp /workspace/data/nnUNet_raw/Dataset001_ImageCAS/splits_final.json \
       /workspace/data/nnUNet_preprocessed/Dataset001_ImageCAS/splits_final.json
```

Trening (v tmux seji):

```bash
tmux new -s nnunet_trening

docker run --gpus 'device=1' --shm-size=32g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    -e nnUNet_raw=/workspace/data/nnUNet_raw \
    -e nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed \
    -e nnUNet_results=/workspace/data/nnUNet_results \
    -e nnUNet_compile=False \
    mednext_izziv nnUNetv2_train 001 3d_fullres 0

# Po dosegu željenega števila epoh: Ctrl+C
# Odlepitev seje: Ctrl+B, nato D
```

### 2. Napoved na testni množici

```bash
docker run --gpus 'device=1' --shm-size=32g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    -e nnUNet_raw=/workspace/data/nnUNet_raw \
    -e nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed \
    -e nnUNet_results=/workspace/data/nnUNet_results \
    -e nnUNet_compile=False \
    mednext_izziv nnUNetv2_predict \
    -i /workspace/data/nnUNet_raw/Dataset001_ImageCAS/imagesTs \
    -o /workspace/predictions/nnunet_test \
    -d 001 -c 3d_fullres -f 0 \
    -chk checkpoint_best.pth
```

### 3. Evalvacija nnU-Net napovedi

```bash
docker run --gpus all --shm-size=16g --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 scripts/eval_nnunet.py \
    --pred_path   /workspace/predictions/nnunet_test \
    --raw_dir     /data/data \
    --output_path /workspace/results/nnunet_metrics.json \
    --model       nnU-Net
```

`eval_nnunet.py` uporablja isti princip kot `run_test.py` — ground truth maske poišče neposredno v surovih podatkih glede na ime napovedane datoteke (`case_XXXX.nii.gz` → `XXXX.label.nii.gz`).

### 4. Kvalitativna primerjava

```bash
docker run --rm \
    -v "$(pwd)":/workspace \
    -v /media/FastDataMama/izziv:/data \
    -w /workspace \
    mednext_izziv python3 scripts/vizualizacija.py \
    --cases      "1,2,3" \
    --raw_dir    /data/data \
    --pred_dir   /workspace/predictions/mednext \
    --output_dir /workspace/results/vizualizacije
```

Skripta za vsak primer shrani sliko s štirimi stolpci: originalna CTA rezina, ground truth maska, napoved modela in analiza napak (True/False Positive/Negative).

## Rezultati (200 naključnih primerov, 120 epoh)

|  Metoda | Dice ↑ | HD95 ↓ | Completeness ↑ | Correctness ↑ | Quality ↑ |
|---------|--------|--------|----------------|---------------|-----------|
| nnU-Net | 0.7470 | 88.96  |     0.8443     |    0.6714     |  0.5963   |
| MedNeXt | 0.6887 | 91.20  |     0.6912     |    0.4963     |  0.4054   |

nnU-Net dosega boljše rezultate na vseh metrikah, kar je pričakovano glede na njegovo avtomatsko optimizacijo arhitekture in preprocessinga za podani dataset. Pri obeh modelih je Completeness višji od Correctness, kar kaže na tendenco po prekomernem napovedovanju (False Positives) — ta vzorec je bolj izrazit pri MedNeXt.

## Reprodukcija rezultatov

Za popolno reprodukcijo zgornjih rezultatov:

1. Zgradite Docker image: `docker build -t mednext_izziv .`
2. Pripravite podatke: `scripts/convert_imagecas_to_nnunet_n_samples.py --n_cases 200 --seed 42`
3. Trenirajte MedNeXt: `run_train.py --epochs 120 --split_id 0`
4. Trenirajte nnU-Net na istem splitu (glejte zgoraj)
5. Evalvirajte oba modela: `run_test.py` in `eval_nnunet.py`
6. Generirajte vizualizacije: `scripts/vizualizacija.py`
