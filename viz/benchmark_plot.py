import numpy as np
import matplotlib.pyplot as plt


from viz.misc import to_latex


class BenchmarkPlot(object):
    def __init__(self):
        pass

    def add(self, data):
        self.data = data

    def plot(self, save_path):
        fig, ax = plt.subplots(nrows=1, ncols=len(self.data), figsize=(3 * len(self.data), 3))
        for i, (k, v) in enumerate(self.data.items()):
            ax[i].set_title(to_latex(k))
            ax[i].plot(np.arange(len(v)), v)
            ax[i].set_xlabel("time")
            ax[i].set_ylabel(to_latex(k))

        fig.tight_layout()
        plt.savefig(save_path)