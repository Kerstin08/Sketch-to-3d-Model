import os.path
import map_generation
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from source.map_generation_dataset import dataset
from source.util import data_type
from source.util import dir_utils

def train(input_dir, output_dir, logs_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient,
        use_generated_model=False, generated_model_path="", devices=1):

    sketch_dir = os.path.join(input_dir, "sketch_mapgen")
    target_dir = os.path.join(input_dir, "target_mapgen")
    if not os.path.exists(sketch_dir) or not os.path.exists(target_dir):
        raise Exception("Sketch dir: {} or target dir: {} does not exists!".format(sketch_dir, target_dir))

    if len(logs_dir) <= 0:
        raise Exception("Logs Path is not given!")
    # Use general folder instead of logs dir since pytorch already takes care of folder versioning.
    dir_utils.create_general_folder(logs_dir)

    if type == "depth":
        given_type = data_type.Type.depth
    elif type == "normal":
        given_type = data_type.Type.normal
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")

    model = map_generation.MapGen(data_type=given_type,
                                  n_critic=n_critic,
                                  batch_size=batch_size,
                                  weight_L1=weight_L1,
                                  gradient_penalty_coefficient=gradient_penalty_coefficient,
                                  output_dir=output_dir,
                                  lr=lr)

    if use_generated_model:
        if not os.path.exists(generated_model_path):
            raise Exception("Generated model paths are not given!")
        model.load_from_checkpoint(generated_model_path,
                                   n_critic=n_critic,
                                   batch_size=batch_size,
                                   weight_L1=weight_L1,
                                   gradient_penalty_coefficient=gradient_penalty_coefficient,
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

    dataSet_train = dataset.DS(True, given_type, sketch_train_dir, target_train_dir)
    dataSet_val = dataset.DS(True, given_type, sketch_val_dir, target_val_dir)

    strategy = None
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'gpu' and (devices > 1 or (devices == -1 and torch.cuda.device_count() > 1)):
        strategy = 'ddp'
    elif accelerator == 'cpu' and (devices != 1):
        raise Exception("Training with mulitple cpus not permitted!")

    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      max_epochs=epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      precision=16,
                      strategy=strategy)
    dataloader_train = DataLoader(dataSet_train, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    dataloader_vaild = DataLoader(dataSet_val, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    trainer.fit(model, dataloader_train, dataloader_vaild)