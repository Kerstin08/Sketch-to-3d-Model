import os.path

import map_generation
import warnings
import argparse
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from source.mapgen_dataset import DataSet
from pytorch_lightning.callbacks import ModelCheckpoint

def run(train, input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        use_generated_model=False, generated_model_path="", use_comparison=True):


    if len(input_dir) <= 0 or not os.path.exists(input_dir):
        raise Exception("Input directory is not given or does not exist!")

    sketch_dir = os.path.join(input_dir, "sketch_mapgen")

    if type == "depth":
        given_type = map_generation.Type.depth
        target_dir = os.path.join(input_dir, "d_mapgen")
        channels = 1
    elif type == "normal":
        given_type = map_generation.Type.normal
        target_dir = os.path.join(input_dir, "n_mapgen")
        channels = 3
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")

    if not os.path.exists(sketch_dir) or not os.path.exists(target_dir):
        raise Exception("Sketch dir: {} or target dir: {} does not exits!".format(sketch_dir, target_dir))

    if len(output_dir) <= 0:
        raise Exception("Checkpoint Path is not given!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if len(logs_dir) <= 0:
        raise Exception("Logs Path is not given!")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    model = map_generation.MapGen(given_type, n_critic, channels, batch_size, weight_L1, use_comparison, output_dir, lr)
    if use_generated_model:
        if not os.path.exists(generated_model_path):
            raise Exception("Generated model paths are not given!")
        model.load_from_checkpoint(generated_model_path)


    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        save_last=True,
        monitor="global_step",
        mode="max",
        dirpath=output_dir,
        filename="MapGen-{epoch:02d}-{global_step}",
        every_n_train_steps=500
    )
    logger = TensorBoardLogger(logs_dir, name="trainModel")
    # Todo: exchange dataset for test, since there should be no target dir any more
    dataSet = DataSet.DS(sketch_dir, target_dir, given_type)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      max_epochs=epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      log_every_n_steps=10)
    if train:
        train_set_size = int(len(dataSet) * 0.8)
        valid_set_size = len(dataSet) - train_set_size
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = data.random_split(dataSet, [train_set_size, valid_set_size], seed)
        dataloader_train = DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4)
        dataloader_vaild = DataLoader(valid_set, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        trainer.fit(model, dataloader_train, dataloader_vaild)

    else:
        if not use_generated_model:
            warnings.warn("Map generation is called on untrained models!")
        dataloader = DataLoader(dataSet, batch_size=1,
                                shuffle=False, num_workers=0)
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
        args.generated_model_path,
        args.use_comparison)

def main(args):
    parser = argparse.ArgumentParser(prog="mapgen_dataset")
    parser.add_argument("--train", type=bool, default=True, help="Train or test")
    parser.add_argument("--input_dir", type=str, default="..\\..\\resources\\sketch_meshes", help="Directory where the input sketches for training are stored")
    parser.add_argument("--output_dir", type=str, default="checkpoint", help="Directory where the checkpoints or the test output is stored")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory where the logs are stored")
    parser.add_argument("--type", type=str, default="normal", help="use \"normal\" or \"depth\" in order to train\\generate depth or normal images")
    parser.add_argument("--epochs", type=int, default=10, help="# of epoch")
    parser.add_argument("--lr", type=float, default=100, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="# of epoch")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=50, help="L1 weight")
    parser.add_argument("--use_generated_model", type=bool, default=False, help="If models are trained from scratch or already trained models are used")
    parser.add_argument("--generated_model_path", type=str, default="..\\..\\output\\test.ckpt", help="If test is used determine if comparison images should be generated")
    parser.add_argument("--use_comparison", type=bool, default=True, help="If test is used determine if comparison images should be generated")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '0_2000_normal',
        '--type', 'normal',
        '--epochs', '100',
        '--lr', '2e-4'
    ]
    main(params)