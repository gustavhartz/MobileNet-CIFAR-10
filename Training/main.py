
import torch
from data import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from lightning_module import LightningModel
import pytorch_lightning as pl
import time
from helpers import str2bool, set_seed
import argparse


def main(args):
    # if needed
    hparams = vars(args)

    set_seed(args)
    pytorch_model = torch.hub.load(
        'pytorch/vision:v0.11.0',
        'mobilenet_v2',
        pretrained=True)

    pytorch_model.classifier[-1] = torch.nn.Linear(
        in_features=1280, out_features=10)

    data = DataModule(args)

    lightning_model = LightningModel(
        pytorch_model, learning_rate=args.lr)

    callbacks = [ModelCheckpoint(
        save_top_k=args.save_top_k, mode='max', monitor="valid_acc")]  # save top 1 model
    logger = CSVLogger(save_dir="logs/", name="my-model")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        log_every_n_steps=100)

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data)
    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    # Learning rate
    argparser.add_argument('--lr', type=float,
                           default=1e-4, help='Learning rate')
    # Batch size
    argparser.add_argument('--batch_size', type=int,
                           default=8, help='Batch size')
    # Seed
    argparser.add_argument('--seed', type=int, default=42, help='Seed')

    # Save top k models
    argparser.add_argument('--save_top_k', type=int, default=-1,
                           help='Top model to save defined by metric. -1 is none')

    # Num epochs
    argparser.add_argument('--num_epochs', type=int,
                           default=10, help='Number of epochs to run training')

    # Num workers
    argparser.add_argument('--num_workers', type=int,
                           default=2, help='Number of dataset workers')

    # Test model
    argparser.add_argument('--test_model', type=str2bool, nargs='?',
                           const=True, default=False, help='Test model. This will not train the model and only run a single evaluation on the predict file using the CUAD metrics')

    args = argparser.parse_args()

    main(args)
