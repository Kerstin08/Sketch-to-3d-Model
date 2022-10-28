import os.path

import map_generation
import warnings
import argparse
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from source.mapgen_dataset import dataset
from pytorch_lightning.callbacks import ModelCheckpoint


def run(train_b, input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        use_generated_model=False, generated_model_path=""):
    if train_b:
        train(input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        use_generated_model, generated_model_path)
    else:
        test(input_dir, output_dir, type, batch_size, generated_model_path)

def train(input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        use_generated_model=False, generated_model_path=""):

    sketch_dir = os.path.join(input_dir, "sketch_mapgen")
    target_dir = os.path.join(input_dir, "target_mapgen")
    if not os.path.exists(sketch_dir) or not os.path.exists(target_dir):
        raise Exception("Sketch dir: {} or target dir: {} does not exists!".format(sketch_dir, target_dir))

    if len(output_dir) <= 0:
        raise Exception("Checkpoint Path is not given!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if len(logs_dir) <= 0:
        raise Exception("Logs Path is not given!")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    model = map_generation.MapGen(n_critic=n_critic,
                                  batch_size=batch_size,
                                  weight_L1=weight_L1,
                                  output_dir=output_dir,
                                  lr=lr)

    if use_generated_model:
        if not os.path.exists(generated_model_path):
            raise Exception("Generated model paths are not given!")
        model.load_from_checkpoint(generated_model_path,
                                   n_critic=n_critic,
                                   batch_size=batch_size,
                                   weight_L1=weight_L1,
                                   output_dir=output_dir,
                                   lr=lr)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=output_dir,
        filename="MapGen-{epoch:02d}-{val_loss}",
    )
    logger = TensorBoardLogger(logs_dir, name="trainModel")

    sketch_train_dir = os.path.join(sketch_dir, "train")
    if not os.path.exists(sketch_train_dir):
        raise Exception("Train dir in {} does not exist".format(sketch_dir))
    sketch_val_dir = os.path.join(sketch_dir, "val")
    if not os.path.exists(sketch_val_dir):
        raise Exception("Val dir in {} does not exist".format(sketch_dir))
    target_train_dir = os.path.join(target_dir, "train")
    if not os.path.exists(target_train_dir):
        raise Exception("Train dir in {} does not exist".format(target_dir))
    target_val_dir = os.path.join(target_dir, "val")
    if not os.path.exists(target_val_dir):
        raise Exception("Val dir in {} does not exist".format(target_dir))

    dataSet_train = dataset.DS(True, sketch_train_dir, target_train_dir)
    dataSet_val = dataset.DS(True, sketch_val_dir, target_val_dir)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=4,
                      max_epochs=epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      precision=16,
                      strategy="ddp")
    dataloader_train = DataLoader(dataSet_train, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    dataloader_vaild = DataLoader(dataSet_val, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    trainer.fit(model, dataloader_train, dataloader_vaild)

def test(input_dir, output_dir,
        type, batch_size, generated_model_path=""):

    if len(input_dir) <= 0 or not os.path.exists(input_dir):
        raise Exception("Input directory is not given or does not exist!")

    sketch_dir = os.path.join(input_dir, "sketch_mapgen")
    if not os.path.exists(sketch_dir):
        raise Exception("Sketch dir: {} does not exists!".format(sketch_dir))

    test_dir = os.path.join(sketch_dir, "test")
    if not os.path.exists(test_dir):
        raise Exception("Test dir in {} does not exist".format(sketch_dir))

    if len(output_dir) <= 0:
        raise Exception("Output Path is not given!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(generated_model_path):
        raise Exception("Generated model paths are not given!")
    model = map_generation.MapGen.load_from_checkpoint(generated_model_path,
                                  batch_size=batch_size,
                                  output_dir=output_dir)


    dataSet = dataset.DS(False, test_dir)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1)
    dataloader = DataLoader(dataSet, batch_size=1,
                                shuffle=False, num_workers=4)
    trainer.test(model, dataloaders=dataloader)


def diff_args(args):
    run(args.train,
        args.input_dir,
        args.output_dir,
        args.logs_dir,
        args.type,
        args.epochs,
        args.lr,
        args.batch_size,
        args.n_critic,
        args.weight_L1,
        args.use_generated_model,
        args.generated_model_path)


def main(args):
    parser = argparse.ArgumentParser(prog="mapgen_dataset")
    parser.add_argument("--train", type=bool, default=True, help="Train or test")
    parser.add_argument("--input_dir", type=str, default="..\\..\\resources\\sketch_meshes",
                        help="Directory where the input sketches for training are stored")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory where the checkpoints or the test output is stored")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory where the logs are stored")
    parser.add_argument("--type", type=str, default="normal",
                        help="use \"normal\" or \"depth\" in order to train\\generate depth or normal images")
    parser.add_argument("--epochs", type=int, default=10, help="# of epoch")
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=50000, help="L1 weight")
    parser.add_argument("--use_generated_model", type=bool, default=False,
                        help="If models are trained from scratch or already trained models are used")
    parser.add_argument("--generated_model_path", type=str, default="..\\..\\output\\test.ckpt",
                        help="If test is used determine if comparison images should be generated")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--input_dir', 'datasets/su_dataset',
        '--type', 'normal',
        '--epochs', '200',
        '--lr', '5e-5',
        '--output_dir', "checkpoint/176_output",
        '--generated_model_path', "checkpoint/MapGen-epoch=176-val_loss=0.08904778212308884.ckpt"
    ]
    main(params)
