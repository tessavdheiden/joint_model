import imageio


class Video(object):
    def __init__(self):
        self.frames = []

    def add(self, f):
        self.frames.append(f)

    def save(self, path):
        imageio.mimsave(path, self.frames, fps=30)