import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

class MLP(pl.LightningModule):
    def __init__(self, mlp_configs, optimizer_configs):
        super().__init__()
        self.save_hyperparameters()
        self.layers = [nn.Linear(mlp_configs["in_channel"], mlp_configs["hidden_channels"][0])]
        self.layers.append(nn.ReLU())
        for h in range(len(mlp_configs["hidden_channels"])-1):
            self.layers.append(nn.Linear(mlp_configs["hidden_channels"][h], mlp_configs["hidden_channels"][h+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(mlp_configs["hidden_channels"][-1], mlp_configs["out_channel"]))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)

        self.optimizer_configs = optimizer_configs
        self.valid_accuracy = tm.Accuracy()
        self.valid_precision = tm.Precision()
        self.valid_recall = tm.Recall()
        self.valid_f1 = tm.F1()
        self.test_accuracy = tm.Accuracy()
        self.test_precision = tm.Precision()
        self.test_recall = tm.Recall()
        self.test_f1 = tm.F1()


    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.binary_cross_entropy(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
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

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), **self.optimizer_configs)