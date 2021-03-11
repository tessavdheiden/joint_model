import numpy as np
import matplotlib.pyplot as plt


from viz.misc import to_latex


class SelectionPlot(object):
    def __init__(self):
        pass

    def add(self, xy, z):
        self.xy = xy
        self.z = z

    def plot(self, save_path, xmin, xmax, ymin, ymax):

        xy_merged = {}
        z_merged = {}
        for z_val, traj in zip(self.z, self.xy):
            key = (int(traj[0, 0] * 100), int(traj[0, 1] * 100))
            if key in xy_merged:
                xy_merged[key].append(traj)
                z_merged[key].append(z_val)
            else:
                xy_merged[key] = [traj]
                z_merged[key] = [z_val]

        d, e = {}, {}
        for key, traj_lst in xy_merged.items():
            d[key] = np.array(traj_lst).mean(0)
            e[key] = np.array(z_merged[key]).mean(0)

        c = plt.cm.plasma(np.linspace(0, 1, len(e)))
        idx = np.argsort(list(e.values()))

        for i, (key, traj) in enumerate(d.items()):
            if idx[i] == 0:
                plt.scatter(traj[:, 0], traj[:, 1], color=c[idx[i]], label=f'min={e[key]:.2f}')
            elif idx[i] == max(idx):
                plt.scatter(traj[:, 0], traj[:, 1], color=c[idx[i]], label=f'max={e[key]:.2f}')
            else:
                plt.scatter(traj[:, 0], traj[:, 1], color=c[idx[i]])
        plt.axis([xmin, xmax, ymin, ymax])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}.png')
        plt.close()