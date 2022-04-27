from imutils.video import FileVideoStream

class ImutilsVideoReader:
    def __init__(self, filename):
        self.video = FileVideoStream(filename)

    def read(self):
        frames = []
        self.video.start()
        while self.video.running():
            frame = self.video.read()
            if frame is not None:
                frames.append(frame)
            else:
                print("Found a null frame")
        self.video.stop()
        return frames