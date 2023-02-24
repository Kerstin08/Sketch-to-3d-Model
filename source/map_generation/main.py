import argparse
import sys

from source.map_generation.test import test
from source.map_generation.train import train
from source.util import dir_utils
from source.util import parse
from source.util import data_type


def run(train_b: bool,
        input_dir: str,
        output_dir: str,
        logs_dir: str,
        checkpoint_dir: str,
        input_data_type: data_type.Type,
        epochs: int,
        lr: float,
        batch_size: int,
        n_critic: int,
        weight_L1: int,
        gradient_penalty_coefficient: int,
        log_frequency: int,
        use_generated_model: bool,
        generated_model_path: str,
        devices: int,
        use_shapenet: bool,
        shapenet_train_size: int):
    if len(output_dir) <= 0:
        raise Exception("Checkpoint Path is not given!")
    dir_utils.create_general_folder(output_dir)

    if train_b:
        train(input_dir, output_dir, logs_dir, checkpoint_dir,
              input_data_type, epochs, lr, batch_size, n_critic, weight_L1,
              gradient_penalty_coefficient, log_frequency, use_generated_model, generated_model_path, devices,
              use_shapenet, shapenet_train_size)
    else:
        test(input_dir, output_dir, logs_dir, input_data_type, generated_model_path, 1, use_shapenet)


def diff_args(args):
    run(args.train,
        args.input_dir,
        args.output_dir,
        args.logs_dir,
        args.checkpoint_dir,
        args.input_data_type,
        args.epochs,
        args.lr,
        args.batch_size,
        args.n_critic,
        args.weight_L1,
        args.gradient_penalty_coefficient,
        args.log_frequency,
        args.use_generated_model,
        args.generated_model_path,
        args.devices,
        args.use_shapenet,
        args.shapenet_train_size)


def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--train", type=parse.p_bool, default="True", dest="train",
                        help="If training should be executed, otherwise test is run; use \"True\" or \"False\" as "
                             "parameter")
    parser.add_argument("--input_dir", type=str, default="datasets/mixed_normal",
                        help="Directory where the input sketches for training are stored")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory where the output is stored")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory where the logs are stored")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory where the checkpoints are stored")
    parser.add_argument("--input_data_type", type=parse.p_data_type, default="normal", dest="input_data_type",
                        help="use \"normal\" or \"depth\" in order to train\\generate depth or normal images")
    parser.add_argument("--epochs", type=int, default=10, help="# of epoch")
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=500, help="L1 weight")
    parser.add_argument("--gradient_penalty_coefficient", type=int, default=10, help="gradient penalty coefficient")
    parser.add_argument("--log_frequency", type=int, default=100, help="log frequency for training")
    parser.add_argument("--use_generated_model", type=parse.p_bool, default="False", dest="use_generated_model",
                        help="If models are trained from scratch or already trained models are used; use \"True\" or "
                             "\"False\" as parameter")
    parser.add_argument("--generated_model_path", type=str, default="test.ckpt",
                        help="If test is used determine if comparison images should be generated")
    parser.add_argument("--devices", type=int, default=4,
                        help="Define the number of cpu or gpu devices used")
    parser.add_argument("--use_shapenet", type=parse.p_bool, default="False", dest="use_shapenet",
                        help="If Shapenet dataset is used")
    parser.add_argument("--shapenet_train_size", type=int, default=400,
                        help="usage of # images per class in shapenet dataset in training epoch. "
                             "Needs to be a common multiple of batch_sizes and devices"
                             "# validation is calculated based on this number")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
