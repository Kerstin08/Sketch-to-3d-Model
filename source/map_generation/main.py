import os.path

import map_generation
import warnings
import source.util.save_load_networks as Save_Load_Network
import argparse

def run(train, use_generated_model, input_dir, target_dir, output_path,
        type, epoch, batch_size, n_critic, weight_L1, weight_BCELoss,
        generated_Gen="", generated_Disc="", use_comparison=False):

    if len(input_dir) <= 0 or len(target_dir) <= 0 or not os.path.exists(input_dir) or not os.path.exists(target_dir):
        raise RuntimeError("Input and Target image dirs are not given or do not exist!")

    if type == "depth":
        given_type = map_generation.Type.depth
    elif type == "normal":
        given_type = map_generation.Type.normal
    else:
        raise RuntimeError("Given type should either be \"normal\" or \"depth\"!")

    model = map_generation.MapGen(given_type, epoch, n_critic, weight_L1, weight_BCELoss, batch_size)

    if len(output_path) <= 0:
        raise RuntimeError("Output Path is not given!")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if use_generated_model:
        if len(generated_Gen) <= 0 and len(generated_Disc) <= 0 and os.path.exists(generated_Gen) and os.path.exists(generated_Disc):
            saved_epoch_gen = Save_Load_Network.load_models(model.G, model.optim_G, generated_Gen)
            print("Generator previously trained for " + str(saved_epoch_gen) + "epochs!")
            saved_epoch_disc = Save_Load_Network.load_models(model.D, model.optim_D, generated_Disc)
            print("Discriminator previously trained for " + str(saved_epoch_gen) + "epochs!")
            if saved_epoch_gen != saved_epoch_disc:
                raise RuntimeError("Epochs of given models are not the same!")
        else:
            raise RuntimeError("Generated model paths are not given!")

    if train:
        model.train(input_dir, target_dir, output_path)

    else:
        if not use_generated_model:
            warnings.warn("Map generation is called on untrained models!")
        model.test(input_dir, target_dir, output_path, use_comparison)


def diff_args(args):
    run(args.train,
        args.use_generated_model,
        args.input_dir,
        args.target_dir,
        args.output_dir,
        args.type,
        args.epoch,
        args.batch_size,
        args.n_critic,
        args.weight_L1,
        args.weight_BCELoss,
        args.generated_Gen,
        args.generated_Disc,
        args.use_comparison)

def main(args):
    parser = argparse.ArgumentParser(prog="dataset_generation")
    parser.add_argument("--train", type=bool, default=True, help="If models are trained")
    parser.add_argument("--use_generated_model", type=bool, default=False, help="If models are trained from scratch or already trained models are used")
    parser.add_argument("--input_dir", type=bool, default="..\\..\\resources\\sketch_meshes", help="Directory where the input sketches for training are stored")
    parser.add_argument("--target_dir", type=bool, default="..\\..\\resources\\n_meshes", help="Directory where the normal or depth maps for training are stored")
    parser.add_argument("--output_dir", type=str, default="..\\..\\output", help="Directory where the checkpoints or the test output is stored")
    parser.add_argument("--type", type=str, default="normal", help="use \"normal\" or \"depth\" in order to train\\generate depth or normal images")
    parser.add_argument("--epoch", type=int, default=100, help="# of epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="# of epoch")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=500, help="L1 weight")
    parser.add_argument("--weight_BCELoss", type=int, default=100, help="L1 weight")
    parser.add_argument("--generated_Gen", type=str, help="Directory where the normal or depth maps for training are stored")
    parser.add_argument("--generated_Disc", type=str, help="Directory where the normal or depth maps for training are stored")
    parser.add_argument("--use_comparison", type=bool, help="If test is used determine if comparison images should be generated")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--train', True,
        '--use_generated_model', False,
        '--input_dir', "..\\..\\resources\\sketch_meshes",
        '--target_dir', "..\\..\\resources\\n_meshes"
        '--output_dir', "..\\..\\checkpoints",
        '--type', "normal",
        '--epochs', 100,
        '--batch_size', 1

    ]
    main(params)