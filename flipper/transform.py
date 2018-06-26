import random
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

class PinballFlip(object):
    """ Play a game of Pinball with labeled PIL images and labels

        Apply random vertical flip to image. Overwrite existing label with a boolean
        indicating if the vertical flip was applied.

        Args:
            p (float): probability of image being flipped. Default value is 0.5

    """

    def __init__(self, p=0.5):
        self.p = p
        self._state = None  # 1 for flip, 0 for no flip
        self.countdown = 0

    def _update_get_state(self):
        """ Updates the state with a new random flip every second read """
        if self.countdown == 0:
            self.countdown = 1
            self._state = int(random.random() < self.p)
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


def plot_random_flips(dataset, sample_index=0, ncols=4):
    """ Plot element of dataset multiple times to capture random effects.
        Only first channel of image is plotted.

        Args:
            dataset (torchvision.Dataset with boolean labels)
                sample_index index of element in dataset
            image (TorchTensor)
                Tensor of shape (C, H, W))

    """
    fig, axes = plt.subplots(ncols=ncols, figsize=(16, 4))
    for ax in axes:
        image, label = dataset[sample_index]
        ax.imshow(np.array(image)[0])
        label = 'yes' if label else 'no'
        ax.set_title(f'Flipped: {label}');