import argparse
from source.map_generation.test import test
from source.map_generation.train import train
from source.util import dir_utils
from source.util import bool_parse


def run(train_b_str, input_dir, output_dir, logs_dir, checkpoint_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient, log_frequency,
        use_generated_model_str, generated_model_path, devices):
    if len(output_dir) <= 0:
        raise Exception("Checkpoint Path is not given!")
    dir_utils.create_general_folder(output_dir)

    train_b = bool_parse.parse(train_b_str)
    use_generated_model = bool_parse.parse(use_generated_model_str)

    if train_b:
        train(input_dir, output_dir, logs_dir, checkpoint_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient, log_frequency, use_generated_model, generated_model_path, devices)
    else:
        test(input_dir, output_dir, logs_dir, type, generated_model_path)

def diff_args(args):
    run(args.train,
        args.input_dir,
        args.output_dir,
        args.logs_dir,
        args.checkpoint_dir,
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
    parser.add_argument("--train", type=str, default="True", help="If training should be executed, otherwise test is run; use \"True\" or \"False\" as parameter")
    parser.add_argument("--input_dir", type=str, default="..\\..\\resources\\sketch_meshes",
                        help="Directory where the input sketches for training are stored")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory where the output is stored")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory where the logs are stored")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory where the checkpoints are stored")
    parser.add_argument("--type", type=str, default="normal",
                        help="use \"normal\" or \"depth\" in order to train\\generate depth or normal images")
    parser.add_argument("--epochs", type=int, default=10, help="# of epoch")
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=500, help="L1 weight")
    parser.add_argument("--gradient_penalty_coefficient", type=int, default=10, help="gradient penalty coefficient")
    parser.add_argument("--log_frequency", type=int, default=15, help="log frequency for training")
    parser.add_argument("--use_generated_model", type=str, default="False",
                        help="If models are trained from scratch or already trained models are used; use \"True\" or \"False\" as parameter")
    parser.add_argument("--generated_model_path", type=str, default="..\\..\\output\\test.ckpt",
                        help="If test is used determine if comparison images should be generated")
    parser.add_argument("--devices", type=int, default=4,
                        help="Define the number of cpu or gpu devices used")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--input_dir', 'datasets/mixed_normal',
        '--type', 'normal',
        '--epochs', '3000',
        '--lr', '5e-5',
        '--output_dir', "out_normal",
        "--batch_size", "83"
#        '--generated_model_path', "checkpoints/last.ckpt"
    ]
    main(params)
