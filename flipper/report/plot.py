import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def plot_random_flips(dataset, sample_index=0, ncols=4):
    """ Plot element of dataset multiple times to capture random effects.
        Only first channel of image is plotted.

        Note: does not return figure. Renders automatically in Jupyter notebook
        with %matplotlib inline

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

def moving_average(losses, alpha=0.1):
    ts = pd.Series(np.array(losses))
    return ts.ewm(alpha=0.1).mean().plot()

