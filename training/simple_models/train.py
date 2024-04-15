import os
import torch
import torch.utils.data as du
from torch.utils.data import random_split
import lightning as L
import yaml
from simple_models.dataset import WildFireData
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

    torch.manual_seed(42)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accuracy_metric = tm.Accuracy(task="binary").to(device)
    precision_metric = tm.Precision(task="binary").to(device)
    recall_metric = tm.Recall(task="binary").to(device)
    f1_metric = tm.F1Score(task="binary").to(device)

    print("Predicting No fires...")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for batch in test_loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        labels = labels.reshape(-1, features.shape[2], features.shape[3])
        pred = torch.zeros_like(labels)
        accuracy_list.append(accuracy_metric(pred, labels))
        precision_list.append(precision_metric(pred, labels))
        recall_list.append(recall_metric(pred, labels))
        f1_list.append(f1_metric(pred, labels))
    print("Accuracy Score: ", torch.mean(torch.tensor(accuracy_list)))
    print("Precision Score: ", torch.mean(torch.tensor(precision_list)))
    print("recall Score: ", torch.mean(torch.tensor(recall_list)))
    print("F1 Score: ", torch.mean(torch.tensor(f1_list)))
        
    print("Predicting Previous Time Step Fire as Fire...")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for batch in test_loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        labels = labels.reshape(-1, features.shape[2], features.shape[3])
        pred = features[:,-1]
        accuracy_list.append(accuracy_metric(pred, labels))
        precision_list.append(precision_metric(pred, labels))
        recall_list.append(recall_metric(pred, labels))
        f1_list.append(f1_metric(pred, labels))
    print("Accuracy Score: ", torch.mean(torch.tensor(accuracy_list)))
    print("Precision Score: ", torch.mean(torch.tensor(precision_list)))
    print("recall Score: ", torch.mean(torch.tensor(recall_list)))
    print("F1 Score: ", torch.mean(torch.tensor(f1_list)))

    print("Predicting Simple Disease Spread...")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for batch in test_loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        labels = labels.reshape(-1, features.shape[2], features.shape[3])
        fire = features[:,-1]
        fire = fire.reshape(-1, 1, fire.shape[1], fire.shape[2])
        conv_kernel = torch.ones(1, 1, 3, 3).to(device)
        pred = F.conv2d(fire, conv_kernel, padding=1)
        pred = torch.where(pred > 0, torch.tensor(1).to(device), torch.tensor(0).to(device))

        pred = pred.squeeze(dim=1)
        accuracy_list.append(accuracy_metric(pred, labels))
        precision_list.append(precision_metric(pred, labels))
        recall_list.append(recall_metric(pred, labels))
        f1_list.append(f1_metric(pred, labels))
    print("Accuracy Score: ", torch.mean(torch.tensor(accuracy_list)))
    print("Precision Score: ", torch.mean(torch.tensor(precision_list)))
    print("recall Score: ", torch.mean(torch.tensor(recall_list)))
    print("F1 Score: ", torch.mean(torch.tensor(f1_list)))


def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--config', dest='config', default='simple_models/configs/config.yaml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
    






