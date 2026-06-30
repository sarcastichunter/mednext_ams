#!/bin/bash
# run_all_folds.sh
# Zažene trening in evalvacijo MedNeXt modela na vseh 4 foldih (4-fold cross-validation).
# Vsak fold ustreza enemu Excel stolpcu: Split-1 -> fold 0, Split-2 -> fold 1, itd.
#
# Uporaba:
#   bash run_all_folds.sh <data_dir> <raw_dir> <save_dir> <epochs> <device>
#
# Primer:
#   bash run_all_folds.sh \
#       data/nnUNet_raw/Dataset001_ImageCAS \
#       /data/data \
#       results \
#       120 \
#       cuda

set -e  # Ustavi ob napaki

DATA_DIR=${1:-"data/nnUNet_raw/Dataset001_ImageCAS"}
RAW_DIR=${2:-"/data/data"}
SAVE_DIR=${3:-"results"}
EPOCHS=${4:-120}
DEVICE=${5:-"cuda"}

echo "============================================"
echo " MedNeXt — 4-fold cross-validation"
echo " Data:    $DATA_DIR"
echo " Raw dir: $RAW_DIR"
echo " Output:  $SAVE_DIR"
echo " Epochs:  $EPOCHS"
echo " Device:  $DEVICE"
echo "============================================"

for FOLD in 0 1 2 3; do
    echo ""
    echo "--------------------------------------------"
    echo " Fold $FOLD / 3"
    echo "--------------------------------------------"

    FOLD_SAVE="$SAVE_DIR/fold_$FOLD/checkpoints"
    METRICS_OUT="$SAVE_DIR/fold_$FOLD/metrics.json"

    mkdir -p "$FOLD_SAVE"

    echo "[Fold $FOLD] Trening..."
    python3 run_train.py \
        --data_dir   "$DATA_DIR" \
        --epochs     "$EPOCHS" \
        --batch_size 1 \
        --lr         1e-4 \
        --save_dir   "$FOLD_SAVE" \
        --split_id   "$FOLD" \
        --device     "$DEVICE"

    # Poišči najboljši checkpoint v tem foldu (najnižji loss v imenu)
    BEST_CKPT=$(ls -t "$FOLD_SAVE"/*.pt 2>/dev/null | head -1)

    if [ -z "$BEST_CKPT" ]; then
        echo "[Fold $FOLD] NAPAKA: Checkpoint ni bil najden v $FOLD_SAVE"
        exit 1
    fi

    echo "[Fold $FOLD] Evalvacija checkpointa: $BEST_CKPT"
    python3 run_test.py \
        --model_path  "$BEST_CKPT" \
        --data_path   "$DATA_DIR" \
        --raw_dir     "$RAW_DIR" \
        --output_path "$METRICS_OUT" \
        --device      "$DEVICE"

    echo "[Fold $FOLD] Metrike shranjene: $METRICS_OUT"
done

echo ""
echo "============================================"
echo " Vsi 4 foldi končani."
echo " Rezultati so v: $SAVE_DIR/fold_*/metrics.json"
echo "============================================"