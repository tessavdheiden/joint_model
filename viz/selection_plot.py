import numpy as np
import matplotlib.pyplot as plt


from viz.misc import to_latex


class SelectionPlot(object):
    def __init__(self):
        pass

    def add(self, xy, z):
        self.xy = xy
        self.z = z

    def plot(self, save_path):
        m = np.argmin(self.z)
        M = np.argmax(self.z)

        low = self.xy[m, :]
        high = self.xy[M, :]
        plt.scatter(low[:, 0], low[:, 1], label='min')
        plt.scatter(high[:, 0], high[:, 1], label='max')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}.png')