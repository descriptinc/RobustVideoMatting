import pims
import numpy as np

class PimsVideoReader:
    def __init__(self, filename):
        self.video = pims.PyAVVideoReader(filename)

    def read(self):
        n_frames = len(self.video)
        frames = []
        for i in range(n_frames):
            # f = video[i].asnumpy()
            f = np.asarray(self.video[i])
            frames.append(f)
        return frames
