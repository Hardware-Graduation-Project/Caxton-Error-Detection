if __name__ == "__main__":
    import os
    import argparse
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint
    from data.data_module import ParametersDataModule
    from model.network_module import ParametersClassifier
    from train_config import *

    # --- Argument parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", default=1234, type=int, help="Set seed for training"
    )
    parser.add_argument(
        "-e", "--epochs", default=MAX_EPOCHS, type=int, help="Number of epochs"
    )
    args = parser.parse_args()
    seed = args.seed

    # --- Set seed for reproducibility ---
    set_seed(seed)

    # --- Logging setup ---
    logs_dir = f"logs/logs-{DATE}/{seed}/"
    logs_dir_default = os.path.join(logs_dir, "default")
    make_dirs(logs_dir)
    make_dirs(logs_dir_default)

    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{DATE}/{seed}/",
        filename=f"MHResAttNet-{DATASET_NAME}-{DATE}" + "-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min",
    )

    # --- Model ---
    model = ParametersClassifier(
        num_classes=3,
        lr=INITIAL_LR,
        transfer=False,
    )

    # --- Data ---
    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
    )

    # --- Force data setup and print sizes ---
   # --- Force data setup and print sizes ---
    data.setup(stage="fit")
    data.setup(stage="test")

    print(f"📊 Dataset sizes → Train: {len(data.train_dataset)}, Val: {len(data.val_dataset)}, Test: {len(data.test_dataset)}")

    # ✅ Debug one batch to confirm data is loading
    print("🔍 Checking one training batch...")
    train_loader = data.train_dataloader()
    for batch in train_loader:
        print("✅ Got a training batch!")
        print("Image shape:", batch[0].shape)
        print("Target shape:", batch[1].shape)
        break

    # --- Trainer ---
    trainer = pl.Trainer(
     num_nodes=1,
     accelerator="cpu",  # or "gpu" if you have one
     devices=1,
     max_epochs=args.epochs,
     logger=tb_logger,
     precision=32,
     callbacks=[checkpoint_callback],
     num_sanity_val_steps=2 if len(data.val_dataset) > 0 else 0,
   )


    # --- Train ---
    trainer.fit(model, data)