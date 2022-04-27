import time
import numpy as np
import pandas as pd
import importlib 
import data_utils

path = "test_fps30_res720_t1m.mp4"

start_time = time.time()
# video = DecordVideoReader(path)
# video = PimsVideoReader(path)
# video = OpenCV2Reader(path)
# video = ImutilsVideoReader(path)

import importlib

video_reading_modules = ["DecordVideoReader", "PimsVideoReader", "OpenCV2Reader", "ImutilsVideoReader", "AVVideoReader"]
data = []
for video_reading_module in video_reading_modules:
    video_reading_class = getattr(importlib.import_module("data_utils"), video_reading_module)
    start_time = time.time()
    video = video_reading_class(path)
    frames = video.read()
    n_frames = len(frames)
    reading_time = time.time()

    m = np.array([x.mean() for x in frames]).mean()
    final_time = time.time()
    profile = {}
    profile["library"] = video_reading_module
    profile["n frames"] = n_frames
    profile["reading fps"] = int(n_frames / (reading_time - start_time))
    profile["Total fps"] = int(n_frames /  (final_time - start_time))
    profile["Read time"] = reading_time - start_time
    profile["Processing time"] = final_time - reading_time
    profile["Total time"] = final_time - start_time
    data.append(profile)

data_df = pd.DataFrame(data=data)
data_df.to_csv("./video_reader_profiling.csv")
print(data_df)