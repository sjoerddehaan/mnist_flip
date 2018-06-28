import numpy.random as random
import torch
import torchvision
from torchvision import transforms
import pathlib

class PinballFlip(object):
    """ Play a game of Pinball with labeled PIL images and labels

        Apply random vertical flip to image. Overwrite existing label with a boolean
        indicating if the vertical flip was applied.

        Args:
            p (float): probability of image being flipped. Default value is 0.5

    """

    def __init__(self, p=0.5, seed=None):
        self.seed = seed
        self.p = p
        self._state = None  # 1 for flip, 0 for no flip
        self.countdown = 0
        if seed:
            random.seed(seed)
            self._random_state = random.get_state()
        else:
            self._random_state = None

    def _update_get_state(self):
        """ Updates the state with a new random flip every second read """
        if self.countdown == 0:
            self.countdown = 1
            if self.seed:
                random.set_state(self._random_state)
            self._state = int(random.random() < self.p)
            if self.seed:
                self._random_state = random.get_state()
        else:
            self.countdown -= 1
        return self._state

    def image_transform(self, image):
        """ Statefull transformation to be applied at image features.

            Args:
                image (PIL)

            Returns:
                image (PIL) : Randomly flipped in vertical direction=

        """
        state = self._update_get_state()
        image = transforms.functional.vflip(image) if state else image
        return image

    def target_transform(self, label):
        """ Statefull transformation to be applied at target.
        Replace original label with a boolean label.

        Args:
            label

        Returns:
            label: boolean indicating if image was flipped

        """
        return self._update_get_state()


def duplicate_channels(image, n=3):
    """

    Args:
        image (TorchTensor of single monogromatic image
                shape (C, H, W))

    Returns:
      (TorchTensor of single monogromatic image
                shape (n * C, H, W))

    """
    return torch.cat((image, ) * n)


def get_flipped_mnist_datasets(path):
    """ Returns datasets of the randomly flipped MNIST images

    Args:
        path (string) disk location for storing the data set

    Returns:
        tuple (trainingset, validationset)
        in torchvision.dataset format

    """
    flipper_train = PinballFlip(p=0.5)
    flipper_dev = PinballFlip(p=0.5, seed=2)
    image_transforms_train = transforms.Compose(
        [transforms.Resize(size=224),
         flipper_train.image_transform,
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         duplicate_channels
         ])
    image_transforms_dev = transforms.Compose(
        [transforms.Resize(size=224),
         flipper_dev.image_transform,
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         duplicate_channels
         ])
    target_transforms_train = flipper_train.target_transform
    target_transforms_dev = flipper_dev.target_transform
    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                                download=True, transform=image_transforms_train,
                                                target_transform=target_transforms_train)
    valset = torchvision.datasets.MNIST(root=path, train=False,
                                                download=True, transform=image_transforms_dev,
                                                target_transform=target_transforms_dev)

    return trainset, valset


def save_dataset(dataset, path):
    """ Take PIL-image dataset and save it as .jpg images """
    path = pathlib.Path(path)
    i=0
    for image, label in dataset:
        label_path = path / str(label)
        label_path.mkdir(exist_ok=True, parents=True)
        img_path = label_path / f'{i}.jpg'
        image.save(img_path)
        i+=1
    print(f"Saved {i} images to {path}")


def save_flipped_mnist(input_path, output_path):
    """ """
    output_path = pathlib.Path(output_path)
    flipper = PinballFlip(p=0.5, seed=2)
    image_transforms = transforms.Compose(
        [transforms.Resize(size=224),
         flipper.image_transform,
         transforms.ToTensor(),
         duplicate_channels,
         transforms.ToPILImage()
         ])

    target_transforms = flipper.target_transform

    trainset = torchvision.datasets.MNIST(root=input_path, train=True,
                                                download=True, transform=image_transforms,
                                                target_transform=target_transforms)
    devset = torchvision.datasets.MNIST(root=input_path, train=False,
                                                download=True, transform=image_transforms,
                                                target_transform=target_transforms)
    save_dataset(trainset, output_path / 'train')
    save_dataset(devset, output_path / 'dev')