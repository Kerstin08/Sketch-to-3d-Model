import os.path
import source.map_generation.map_generation as map_generation
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
import source.map_generation_dataset.dataset as dataset
import source.util.data_type as data_type

def test(input_dir, output_dir,
        type, generated_model_path, batch_size=1, devices=1):

    if len(input_dir) <= 0 or not os.path.exists(input_dir):
        raise Exception("Input directory: {} is not given or does not exist!".format(input_dir))

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
                                  batch_size=batch_size,
                                  output_dir=output_dir)

    if os.path.exists(test_dir_target):
        dataSet = dataset.DS(False, given_type, test_dir_sketch, test_dir_target)
    else:
        dataSet = dataset.DS(False, given_type, test_dir_sketch)

    strategy = None
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'gpu' and (devices > 1 or (devices == -1 and torch.cuda.device_count() > 1)):
        strategy = 'ddp'
    elif accelerator == 'cpu':
        raise Exception("Training with cpus not permitted!")
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      precision=16,
                      devices=devices,
                      strategy=strategy)
    dataloader = DataLoader(dataSet, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    trainer.test(model, dataloaders=dataloader)
