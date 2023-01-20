import os.path
import source.map_generation.map_generation as map_generation
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from source.map_generation_dataset import dataset
from source.map_generation_dataset import dataset_ShapeNet
from source.util import data_type
from source.util import dir_utils

def test(input_dir, output_dir, logs_dir,
        type, generated_model_path, devices=1,
        shapenet=False):

    if len(input_dir) <= 0 or not os.path.exists(input_dir):
        raise Exception("Input directory: {} is not given or does not exist!".format(input_dir))
    if len(logs_dir) <= 0:
        raise Exception("Logs Path is not given!")
    logs_dir_name = "testModel"
    # Use general folder instead of logs dir since pytorch already takes care of folder versioning.
    dir_utils.create_general_folder(os.path.join(logs_dir, logs_dir_name))
    sketch_dir = os.path.join(input_dir, "sketch_mapgen")
    target_dir = os.path.join(input_dir, "target_mapgen")
    if not os.path.exists(sketch_dir):
        raise Exception("Sketch dir: {} does not exists!".format(sketch_dir))
    test_dir_sketch = os.path.join(sketch_dir, "test")
    test_dir_target = os.path.join(target_dir, "test")

    if type == "depth":
        given_type = data_type.Type.depth
    elif type == "normal":
        given_type = data_type.Type.normal
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")

    if not os.path.exists(generated_model_path):
        raise Exception("Generated model paths are not given!")
    model = map_generation.MapGen.load_from_checkpoint(generated_model_path,
                                  output_dir=output_dir)

    if shapenet and os.path.exists(test_dir_target):
        dataSet = dataset_ShapeNet.DS(False, given_type, test_dir_sketch, test_dir_target, True)
    elif shapenet:
        dataSet = dataset_ShapeNet.DS(False, given_type, test_dir_sketch, True)
    elif os.path.exists(test_dir_target):
        dataSet = dataset.DS(False, given_type, test_dir_sketch, test_dir_target)
    else:
        dataSet = dataset.DS(False, given_type, test_dir_sketch)

    strategy = None
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'gpu' and (devices > 1 or (devices == -1 and torch.cuda.device_count() > 1)):
        strategy = 'ddp'
    elif accelerator == 'cpu':
        raise Exception("Testing with cpus not permitted!")

    # Logging creates a lot of unnecessary folders like version, which in pipeline is handled differently in order
    # to compile the logs of a version in one version folder. However,
    # for solely testing those folders are useful to distinct from training logs and
    # to get current version.
    logger = TensorBoardLogger(logs_dir, name=logs_dir_name)
    trainer = Trainer(accelerator=accelerator,
                      precision=16,
                      devices=devices,
                      strategy=strategy,
                      logger=logger,
                      num_nodes=1)
    dataloader = DataLoader(dataSet, batch_size=1,
                                shuffle=False, num_workers=48)
    trainer.test(model, dataloaders=dataloader)
