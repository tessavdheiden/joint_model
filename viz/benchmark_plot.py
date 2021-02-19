import numpy as np
import matplotlib.pyplot as plt


def to_latex(x):
    s = str(x)
    for i,c in enumerate(s):
        if c.isdigit():
            s = s[:i] + '_' + s[i:]
            break

    K = ['dddot', 'ddot', 'dot']
    V = ['\dddot{', '\ddot{', '\dot{']
    d = dict(zip(K, V))
    for k in K:
        if k in s:
            s = s.replace(k, d[k]) + "}"
            break

    return f'${s}$'


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