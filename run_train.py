import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data.dataset_nnunet import get_dataloaders
from mednext.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from src.trening.train_loop import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNeXt on nnU-Net v2 dataset")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Pot do nnU-Net v2 dataset direktorija (DatasetXXX_Task)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Število epochov za trening")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (3D modeli običajno zahtevajo majhne batch size)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Direktorij za shranjevanje checkpointov")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda ali cpu")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Kateri fold uporabiti (0–3 za 4-fold CV)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Nalaganje podatkov...")
    train_loader, val_loader = get_dataloaders(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        split_id=args.split_id,
    )

    print("Inicializacija modela...")
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

    print("Inicializacija optimizerja in loss funkcije...")
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Inicializacija trenerja...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        save_dir=args.save_dir,
        mixed_precision=True,
    )

    print("Začetek treninga...")
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()