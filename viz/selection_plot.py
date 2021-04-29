import numpy as np
import matplotlib.pyplot as plt


from viz.misc import to_latex


class SelectionPlot(object):
    def __init__(self):
        pass

    def add(self, x):
        self.x = x

    def plot(self, save_path):
        plt.hist(self.x, density=True, bins=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}.png')
        plt.close()