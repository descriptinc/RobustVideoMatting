import av
import numpy as np

class AVVideoReader:
    def __init__(self, filename):
        self.video = av.open(filename)

    def read(self):
        frames = [np.array(f.to_ndarray()) for f in self.video.decode(0)]
        return frames