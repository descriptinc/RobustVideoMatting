import decord

class DecordVideoReader:
    def __init__(self, filename):
        self.video = decord.VideoReader(filename)
    
    def read(self):
        n_frames = len(self.video)
        frames = []
        for i in range(n_frames):
            f = self.video[i].asnumpy()
            frames.append(f)
        return frames