import os.path
import map_generation
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from source.map_generation_dataset import dataset
from source.util import data_type

def test(input_dir, output_dir,
        type, generated_model_path=""):

    if len(input_dir) <= 0 or not os.path.exists(input_dir):
        raise Exception("Input directory is not given or does not exist!")

    sketch_dir = os.path.join(input_dir, "sketch_mapgen")
    target_dir = os.path.join(input_dir, "target_mapgen")
    if not os.path.exists(sketch_dir) or not os.path.exists(target_dir):
        raise Exception("Sketch dir: {} or target dir: {} does not exists!".format(sketch_dir, target_dir))

    test_dir = os.path.join(sketch_dir, "test")
    if not os.path.exists(test_dir):
        raise Exception("Test dir in {} does not exist".format(sketch_dir))

    if len(output_dir) <= 0:
        raise Exception("Output Path is not given!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if type == "depth":
        given_type = data_type.Type.depth
    elif type == "normal":
        given_type = data_type.Type.normal
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")

    if not os.path.exists(generated_model_path):
        raise Exception("Generated model paths are not given!")
    model = map_generation.MapGen.load_from_checkpoint(generated_model_path,
                                  batch_size=1,
                                  output_dir=output_dir)


    dataSet = dataset.DS(False, given_type, test_dir, target_dir)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1)
    dataloader = DataLoader(dataSet, batch_size=1,
                                shuffle=False, num_workers=4)
    trainer.test(model, dataloaders=dataloader)
