import torch
import torch.autograd as autograd
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import flipper.report as report


class ModelTrainer(object):
    """ """
    def __init__(self, model, trainset, devset, args):
        """
        Args:
            model (toch.nn)
            trainset (tochvision.dataset)
            devset (torchvision.dataset)
            args (dict) contains arguments like batchsize, learning rate
        """
        self.args = args
        self.n_batch = 0
        self.n_epoch = 0
        self.train_losses = []
        self.model = model
        self.trainset = trainset
        self.devset = devset
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True)
        self._create_devloader()
        self.trainiter = iter(self.trainloader)
        self.optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.model = self.model.cuda()


    def train_batch(self, verbose=False):
        """ Train model for one epoch and return average loss """
        try:
            batch = next(self.trainiter)
        except StopIteration:
            self.trainiter = iter(self.trainloader)
            batch = next(self.trainiter)
            self.n_batch += 1
        x, y = batch
        if self.args.cuda:
            x, y = x.cuda(), y.cuda()
        self.n_epoch += 1
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        self.train_losses.append(loss.cpu().data)
        if verbose:
            print(f"batch: {self.n_epoch} / {self.n_batch} loss: {loss}")

    def evaluate_dev(self, n_batches=20):
        """ Evaluate model on devset """
        losses = []
        accs = []
        self._create_devloader()
        batches = iter(self.devloader)
        n_batches = min(n_batches, len(self.devset))
        self.model.eval()
        for i in tqdm(range(n_batches)):
            x, y = next(batches)
            if self.args.cuda:
                x, y = x.cuda(), y.cuda()
            scores = self.model(x)
            predictions = torch.argmax(scores, 1)
            correct = (predictions == y).double()
            acc = torch.mean(correct)
            accs.append(acc.cpu().data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(scores, y)
            losses.append(loss.cpu().data)
        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        scores = {'loss': avg_loss,
                  'acc': avg_acc}
        return scores

    def predict_(self, batch):
        pass

    def _create_devloader(self):
        self.devloader = torch.utils.data.DataLoader(
            self.devset,
            batch_size=4,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True)

    def plot_losses(self, alpha=0.1):
        if self.train_losses:
            report.plot.moving_average(self.train_losses, alpha=alpha)

def train_model(train_data, dev_data, model, args):
    """ Train model for multiple epochs

    Args:
        train_data (tochvision.dataset)
        dev_data (torchvision.dataset)
        model (toch.nn)
        args (dict) contains arguments like batchsize, learning rate


    """

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    devloader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print("-------------\nEpoch {}:\n".format(epoch))


        loss = run_epoch(trainloader, True, model, optimizer, args)

        print('Train loss: {:.6f}'.format( loss))

        print()

        val_loss = run_epoch(devloader, False, model, optimizer, args)
        print('Val loss: {:.6f}'.format( val_loss))

        # Save model
        torch.save(model, args.save_path)


def run_epoch(dataloader, is_training, model, optimizer, args):
    """ Train model for one epoch and return average loss """
    losses = []
    batch = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for x, y in tqdm(dataloader):
        batch += 1
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        if is_training:
            optimizer.zero_grad()

        out = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data)
        print(f"batch: {batch} loss: {loss}")

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss