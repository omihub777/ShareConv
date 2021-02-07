import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import sys, os
# sys.path.append(os.path.abspath("/"))
from utils import get_dataset, get_model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="c10", type=str)
parser.add_argument("--model", default="spreact18", type=str, help="[spreact18, preact18]")
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--learning-rate", default=1e-1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--max_epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--precision", default=32, type=int)
parser.add_argument("--api-key", required=True)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--milestones", default=[100, 150], nargs="+", type=int)
parser.add_argument("--weight-decay", default=1e-4, type=float)
args = parser.parse_args()
args.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
args.num_workers = 8 * args.num_gpus if args.num_gpus else 8

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl =torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        self.model = get_model(hparams)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = self.accuracy(out, label)
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = self.accuracy(out, label)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

if __name__ == "__main__":
    experiment_name = f"{args.model}_{args.dataset}"
    logger = pl.loggers.CometLogger(
        api_key=args.api_key,
        save_dir="logs",
        project_name="share_conv",
        experiment_name=experiment_name
    )
    args.api_key = None # Initialize API Key for privacy.
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.num_gpus, benchmark=True, logger=logger, max_epochs=args.max_epochs, weights_summary="full", progress_bar_refresh_rate=0)
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)
