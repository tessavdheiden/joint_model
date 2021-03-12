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

        c = plt.cm.viridis(np.linspace(0, 1, len(e)))
        e = {k: e[k] for k in sorted(e, key=e.get)}

        for i, (k, z_val) in enumerate(e.items()):
            traj = d[k]
            if i == 0:
                plt.scatter(traj[:, 0], traj[:, 1], color=c[i], label=f'min={z_val:.4f}')
            elif i == len(e) - 1:
                plt.scatter(traj[:, 0], traj[:, 1], color=c[i], label=f'max={z_val:.4f}')
            else:
                plt.scatter(traj[:, 0], traj[:, 1], color=c[i])
        plt.axis([xmin, xmax, ymin, ymax])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}.png')
        plt.close()