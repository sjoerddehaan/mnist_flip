import argparse

parser = argparse.ArgumentParser(description='flip MNIST')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=4, help='number of epochs for train [default: 4]')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for training [default: 4]')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--train', action='store_true', default=False, help='enable train')
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')