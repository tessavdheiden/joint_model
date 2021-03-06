import matplotlib.pyplot as plt

from viz.misc import to_latex


class LandscapePlot(object):
    def __init__(self):
        pass

    def add(self, xy, z):
        self.xy = xy
        self.z = z

    def plot(self, save_path, mi=1, ma=-1):
        names = self.xy.columns.values
        for xname, yname in zip(names[::2], names[1::2]):
            x = self.xy[xname].values
            y = self.xy[yname].values
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4))
            if mi < ma:
                c = ax.hexbin(x, y, gridsize=20, C=self.z[:], mincnt=1, vmin=mi, vmax=ma)
            else:
                c = ax.hexbin(x, y, gridsize=20, C=self.z[:], mincnt=1)
            plt.colorbar(c)
            ax.set_xlabel(to_latex(xname))
            ax.set_ylabel(to_latex(yname))
            #ax.axis('square')
            plt.tight_layout()
            plt.savefig(f'{save_path}_{xname}_vs_{yname}.png')
        plt.close('all')