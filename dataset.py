import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat import *

class Dataset:
    def __init__(self, n_samples=1000, x1_range=(-1, 1), x2_range=(-1, 1), noise_std=0.1):
        """
        Initialize the Dataset class.

        Parameters:
        - n_samples: int, number of samples to generate.
        - x1_range: tuple, range for x1 values (min, max).
        - x2_range: tuple, range for x2 values (min, max).
        - noise_std: float, standard deviation of the noise added to the target values.
        """
        self.n_samples = n_samples
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.noise_std = noise_std
        self.data = None
        self.noiseless_data = None

    def target_function(self, x1, x2):
        """
        Define the target function for generating y values.

        Modify this function to customize the relationship between x1, x2, and y.
        """
        return np.sin(x1) * np.cos(x2) + x1**2 - x2**2

    def generate(self):
        """
        Generate the dataset with noisy and noiseless targets and store them in a pandas DataFrame.
        """
        # Generate x1 and x2
        x1 = np.random.uniform(self.x1_range[0], self.x1_range[1], self.n_samples)
        x2 = np.random.uniform(self.x2_range[0], self.x2_range[1], self.n_samples)

        # Compute y using the target function with added noise
        noise = np.random.normal(0, self.noise_std, self.n_samples)
        t = self.target_function(x1, x2) + noise
        t_noiseless = self.target_function(x1, x2)

        # Store the data
        self.data = pd.DataFrame({'x1': x1, 'x2': x2, 't': t})
        self.noiseless_data = pd.DataFrame({'x1': x1, 'x2': x2, 't': t_noiseless})

    def inputs(self):
        """
        Return the inputs (x1, x2) as a Mat object.
        """
        if self.data is None:
            print("Dataset has not been generated yet. Call the generate() method first.")
            return None
        return Mat(self.data[['x1', 'x2']].values)

    def targets(self):
        """
        Return the targets (y) as a Mat object.
        """
        if self.data is None:
            print("Dataset has not been generated yet. Call the generate() method first.")
            return None
        return Mat(self.data['t'].values.reshape(-1, 1))

    def plot(self):
        """
        Plot the datasets (noisy and noiseless) as a heatmap if it has been generated.
        """
        if self.data is None:
            print("Dataset has not been generated yet. Call the generate() method first.")
            return
        if self.noiseless_data is None:
            print("Noiseless Dataset has not been generated yet. Call the generate() method first.")
            return

        fig = plt.figure()

        plt.scatter(self.data['x1'], self.data['x2'], c=self.data['t'], cmap='plasma', alpha=0.8)
        plt.colorbar(label='t')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('2D Projection of Data')
        plt.show()

        fig = plt.figure()
        plt.scatter(self.noiseless_data['x1'], self.noiseless_data['x2'], c=self.noiseless_data['t'], cmap='plasma', alpha=0.8)
        plt.colorbar(label='t')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('2D Projection of Noiseless Data')
        plt.show()
