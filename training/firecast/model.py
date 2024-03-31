import torch
import torch.nn.functional as F
import lightning as L
# import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

class FireCast(L.LightningModule):
    def __init__(self, firecast_configs, optimizer_configs):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(firecast_configs["in_channels"], 32, kernel_size=3, padding="same"),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
            nn.Flatten(),
        )
        self.optimizer_configs = optimizer_configs
        self.valid_accuracy = tm.Accuracy(task="binary")
        self.valid_precision = tm.Precision(task="binary")
        self.valid_recall = tm.Recall(task="binary")
        self.valid_f1 = tm.F1Score(task="binary")
        self.test_accuracy = tm.Accuracy(task="binary")
        self.test_precision = tm.Precision(task="binary")
        self.test_recall = tm.Recall(task="binary")
        self.test_f1 = tm.F1Score(task="binary")

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.binary_cross_entropy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.binary_cross_entropy(output, target)

        target = target.type(torch.int64)
        output = torch.round(output)
        metrics = {
            "valid_acc": self.valid_accuracy(output, target), 
            "valid_prec": self.valid_precision(output, target),
            "valid_recall": self.valid_recall(output, target), 
            "valid_f1": self.valid_f1(output, target),
        }
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.binary_cross_entropy(output, target)
        target = target.type(torch.int64)
        output = torch.round(output)
        metrics = {
            "test_acc": self.test_accuracy(output, target), 
            "test_prec": self.test_precision(output, target),
            "test_recall": self.test_recall(output, target), 
            "test_f1": self.test_f1(output, target),
        }
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), **self.optimizer_configs)