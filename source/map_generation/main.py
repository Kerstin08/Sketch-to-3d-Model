import argparse
from source.map_generation.test import test
from source.map_generation.train import train
import source.util.dir_utils as dir_utils
import os
import torch

def run(train_b, input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient, log_frequency,
        use_generated_model=False, generated_model_path="", devices=1):
    if len(output_dir) <= 0:
        raise Exception("Checkpoint Path is not given!")
    dir_utils.create_general_folder(output_dir)

    if train_b:
        train(input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient, log_frequency, use_generated_model, generated_model_path, devices)

        # Add teststep with model that had the smallest validation loss
        loss = float("inf")
        for root, _, files in os.walk(output_dir):
            for file in files:
                if "val_loss" in file:
                    val = file.rsplit("val_loss=")[1].rsplit(".ckpt")[0]
                    if float(val) and loss > float(val):
                        loss = float(val)
                        generated_model_path = os.path.join(root, file)
        output_dir_test = os.path.join(output_dir, "test")
        dir_utils.create_general_folder(output_dir_test)
        test(input_dir, output_dir_test, logs_dir, type, generated_model_path)
    else:
        test(input_dir, output_dir, logs_dir, type, generated_model_path)

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
        args.gradient_penalty_coefficient,
        args.log_frequency,
        args.use_generated_model,
        args.generated_model_path,
        args.devices)


def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
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
    parser.add_argument("--weight_L1", type=int, default=500, help="L1 weight")
    parser.add_argument("--gradient_penalty_coefficient", type=int, default=10, help="gradient penalty coefficient")
    parser.add_argument("--log_frequency", type=int, default=15, help="log frequency for training")
    parser.add_argument("--use_generated_model", type=bool, default=False,
                        help="If models are trained from scratch or already trained models are used")
    parser.add_argument("--generated_model_path", type=str, default="..\\..\\output\\test.ckpt",
                        help="If test is used determine if comparison images should be generated")
    parser.add_argument("--devices", type=int, default=4,
                        help="Define the number of cpu or gpu devices used")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--input_dir', 'datasets/mixed_0_2500_depth',
        '--type', 'depth',
        '--epochs', '5',
        '--lr', '9e-5',
        '--output_dir', "checkpoints_depth",
#        '--generated_model_path', "checkpoints/last.ckpt"
    ]
    main(params)
