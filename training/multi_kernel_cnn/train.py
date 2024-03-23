import os
import torch
import torch.utils.data as du
from torch.utils.data import random_split
import lightning as L
import yaml
from multi_kernel_cnn.model import FireCast
from multi_kernel_cnn.dataset import WildFireData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm


def main(yaml_file_dir="configs/config.yaml"):

    #load config file
    cfg = yaml.safe_load(open(yaml_file_dir))

    # load dataset 70/10/20 split
    print("Loading dataset...")
    dataset = WildFireData(**cfg["dataset"])
    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = int(len(dataset) * 0.2)
    test_set_size = len(dataset) - (train_set_size + valid_set_size)
    print("Finished loading dataset.")

    # split the dataset into a train/validation/test set 
    print("Splitting dataset...")
    train_set, valid_set, test_set = random_split(dataset, [train_set_size, valid_set_size, test_set_size])
    print("Finished splitting dataset.")

    print("Creating dataloaders...")
    train_loader = du.DataLoader(train_set, **cfg["train_dataloader"])
    valid_loader = du.DataLoader(valid_set, **cfg["valid_dataloader"])
    test_loader = du.DataLoader(test_set, **cfg["test_dataloader"])
    print("Finished creating dataloaders.")

    # load model
    print("Loading model...")
    model = FireCast(cfg["model"], cfg["optimizer"])
    print("Finished loading model.")

    checkpoint_callback = ModelCheckpoint(**cfg["callbacks"]["model_checkpoint"])

    # train model
    if cfg["logger"] is not None:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = L.Trainer(**cfg["trainer"], logger=logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(**cfg["trainer"], callbacks=[checkpoint_callback])


    print("Training model...")
    trainer.fit(model, train_loader, valid_loader)
    print("Finished training model.")

    print("Testing model...")
    trainer.test(model, test_loader)
    print("Finished testing model.")

def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--config', dest='config', default='multi_kernel_cnn/configs/config.yaml')
    parser.add_argument('--saveckpt', dest='saveckpt', default='training/outputs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
    






