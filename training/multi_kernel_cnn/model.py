import torch
import torch.nn.functional as F
import lightning as L
# import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding="same")
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x1 = self.relu(x1)
        x2 = self.conv3x3(x)
        x2 = self.relu(x2)
        x3 = self.conv5x5(x)
        x3 = self.relu(x3)
        return torch.cat([x1, x2, x3], dim=1)

class FireCast(L.LightningModule):
    def __init__(self, firecast_configs, optimizer_configs):
        super().__init__()
        self.save_hyperparameters()

        encoder = []
        encoder.append(Block(firecast_configs["in_channels"], 16))
        encoder.append(nn.MaxPool2d(2, 2))
        encoder.append(Block(16*3, 32))
        encoder.append(nn.MaxPool2d(2, 2))
        encoder.append(Block(32*3, 64))
        encoder.append(nn.MaxPool2d(2, 2))
        encoder.append(Block(64*3, 128))
        encoder.append(nn.MaxPool2d(2, 2))
        encoder.append(Block(128*3, 256))
        encoder.append(nn.MaxPool2d(2, 2))

        decoder = []
        decoder.append(nn.Flatten())
        decoder.append(nn.Linear(3072, 128))
        decoder.append(nn.ReLU())
        decoder.append(nn.Dropout(firecast_configs["dropout_rate"]))
        decoder.append(nn.Linear(128, 256))
        decoder.append(nn.ReLU())
        decoder.append(nn.Dropout(firecast_configs["dropout_rate"]))
        decoder.append(nn.Linear(256, 4096))
        decoder.append(nn.Sigmoid())
        self.model = nn.Sequential(*encoder, *decoder)

        self.optimizer_configs = optimizer_configs
        self.valid_accuracy = tm.Accuracy(task="binary")
        self.valid_precision = tm.Precision(task="binary")
        self.valid_recall = tm.Recall(task="binary")
        self.valid_f1 = tm.F1Score(task="binary")
        self.test_accuracy = tm.Accuracy(task="binary")
        self.test_precision = tm.Precision(task="binary")
        self.test_recall = tm.Recall(task="binary")
        self.test_f1 = tm.F1Score(task="binary")

        self.threshold = firecast_configs["threshold"]

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        # overlap = torch.sum(output * target, dim=-1)
        # total_area = torch.sum(output, dim=-1) + torch.sum(target, dim=-1)
        # loss = (2 * overlap / total_area) # loss function is the dice coefficient
        # loss = 1 - loss
        # loss = torch.mean(loss)
        loss = F.binary_cross_entropy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        # overlap = torch.sum(output * target, dim=-1)
        # total_area = torch.sum(output, dim=-1) + torch.sum(target, dim=-1)
        # loss = (2 * overlap / total_area) # loss function is the dice coefficient
        # loss = 1 - loss
        # loss = torch.mean(loss)
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
        # overlap = torch.sum(output * target, dim=-1)
        # total_area = torch.sum(output, dim=-1) + torch.sum(target, dim=-1)
        # loss = (2 * overlap / total_area) # loss function is the dice coefficient
        # loss = 1 - loss
        # loss = torch.mean(loss)
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
        return torch.optim.AdamW(self.parameters(), **self.optimizer_configs)