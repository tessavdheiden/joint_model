from collections import namedtuple
import matplotlib.pyplot as plt


Record = namedtuple('Transition', ['episode', 'ε'])


class RecordPlot(object):
    def __init__(self):
        self.records = []

    def add(self, x, y):
        self.records.append(Record(x, y))

    def plot(self, save_path):
        fig, ax = plt.subplots()
        ax.scatter([r[0] for r in self.records], [r[1] for r in self.records])
        ax.set_xlabel(Record._fields[0])
        ax.set_ylabel(Record._fields[1])
        ax.grid('on')
        fig.tight_layout()
        plt.savefig(save_path)
