import os.path
import map_generation
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from source.map_generation_dataset import dataset
from source.map_generation_dataset import dataset_ShapeNet
from source.util import data_type
from source.util import dir_utils
def train(input_dir, output_dir, logs_dir, checkpoint_dir,
        type, epochs, lr, batch_size, n_critic, weight_L1,
        gradient_penalty_coefficient, log_frequency,
        use_generated_model=False, generated_model_path="", devices=1,
        shapenet=False, shapenet_train_size=200):

    sketch_dir = os.path.join(input_dir, "sketch_map_generation")
    target_dir = os.path.join(input_dir, "target_map_generation")
    if not os.path.exists(sketch_dir) or not os.path.exists(target_dir):
        raise Exception("Sketch dir: {} or target dir: {} does not exists!".format(sketch_dir, target_dir))

    logs_dir_name = "trainModel"
    if len(logs_dir) <= 0:
        raise Exception("Logs Path is not given!")
    # Use general folder instead of logs dir since pytorch already takes care of folder versioning.
    dir_utils.create_general_folder(os.path.join(logs_dir, logs_dir_name))

    if type == "depth":
        given_type = data_type.Type.depth
    elif type == "normal":
        given_type = data_type.Type.normal
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")

    model = map_generation.MapGen(data_type=given_type,
                                  n_critic=n_critic,
                                  weight_L1=weight_L1,
                                  gradient_penalty_coefficient=gradient_penalty_coefficient,
                                  output_dir=output_dir,
                                  lr=lr,
                                  batch_size=batch_size)

    if use_generated_model:
        if not os.path.exists(generated_model_path):
            raise Exception("Generated model paths are not given!")
        model.load_from_checkpoint(generated_model_path,
                                   n_critic=n_critic,
                                   weight_L1=weight_L1,
                                   gradient_penalty_coefficient=gradient_penalty_coefficient,
                                   output_dir=output_dir,
                                   lr=lr,
                                   batch_size=batch_size)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename="MapGen-{epoch:02d}-{val_loss}",
    )
    logger = TensorBoardLogger(logs_dir, name=logs_dir_name)

    sketch_train_dir = os.path.join(sketch_dir, "train")
    if not os.path.exists(sketch_train_dir):
        raise Exception("Train dir in {} does not exist".format(sketch_dir))
    sketch_val_dir = os.path.join(sketch_dir, "val")
    if not os.path.exists(sketch_val_dir):
        raise Exception("Val dir in {} does not exist".format(sketch_dir))
    sketch_test_dir = os.path.join(sketch_dir, "test")
    if not os.path.exists(sketch_test_dir):
        raise Exception("Test dir in {} does not exist".format(sketch_dir))
    target_train_dir = os.path.join(target_dir, "train")
    if not os.path.exists(target_train_dir):
        raise Exception("Train dir in {} does not exist".format(target_dir))
    target_val_dir = os.path.join(target_dir, "val")
    if not os.path.exists(target_val_dir):
        raise Exception("Val dir in {} does not exist".format(target_dir))
    target_test_dir = os.path.join(target_dir, "test")
    if not os.path.exists(target_test_dir):
        raise Exception("Test dir in {} does not exist".format(target_dir))

    if shapenet:
        # Compute train, validation split based on ratio used by Kato et al. (Neural mesh renderer)
        split_train, split_val = 87.5, 12.5
        split_train_val = shapenet_train_size/split_train * 100
        shapenet_val_size = int(split_train_val * 12.5 / 100)
        print ("Validation size {0}".format(shapenet_val_size))
        dataSet_train = dataset_ShapeNet.DS(True, given_type, sketch_train_dir, target_train_dir, size=shapenet_train_size, full_ds=False)
        dataSet_val = dataset_ShapeNet.DS(True, given_type, sketch_val_dir, target_val_dir, size=shapenet_val_size, full_ds=False)
        dataSet_test = dataset_ShapeNet.DS(True, given_type, sketch_test_dir, target_test_dir, full_ds=True)
    else:
        dataSet_train = dataset.DS(True, given_type, sketch_train_dir, target_train_dir)
        dataSet_val = dataset.DS(True, given_type, sketch_val_dir, target_val_dir)
        dataSet_test = dataset.DS(True, given_type, sketch_test_dir, target_test_dir)

    strategy = None
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'gpu' and (devices > 1 or (devices == -1 and torch.cuda.device_count() > 1)):
        strategy = 'ddp'
    elif accelerator == 'cpu':
        raise Exception("Training with cpus not permitted!")

    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      max_epochs=epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      precision=16,
                      strategy=strategy,
                      log_every_n_steps=log_frequency)


    dataloader_train = DataLoader(dataSet_train, batch_size=batch_size,
                                  shuffle=True, num_workers=48)
    dataloader_vaild = DataLoader(dataSet_val, batch_size=batch_size,
                                  shuffle=False, num_workers=48)
    trainer.fit(model, dataloader_train, dataloader_vaild)

    dataloader_test = DataLoader(dataSet_test, batch_size=1,
                                  shuffle=False, num_workers=48)
    trainer.test(model, dataloader_test)