from typing import Callable

import numpy as np
import matplotlib

all = ["MPLVideoRenderer"]

class MPLVideoRenderer:
    
    def __init__(self, fig_constructor: Callable, fig_updater: Callable, live=False):
        self.fig_constructor = fig_constructor
        self.fig_updater = fig_updater
        self.live = live
        self.fig_data = None
        self.frames = []
        
        if live:
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")
        
    def _get_frame_rgb(self):
        if self.fig_data is None:
            raise ValueError("No figure data available.")
        fig = self.fig_data["figure"]
        fig.canvas.draw()
        
        if self.live:
            matplotlib.pyplot.pause(0.05)

        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        return frame
    
    def render_frame(self, sim):
        if self.fig_data is None:
            self.fig_data = self.fig_constructor(sim)
        else:
            self.fig_data = self.fig_updater(self.fig_data, sim)
        frame = self._get_frame_rgb()
        self.frames.append(frame)
    
    def to_mp4(self, filename="rllbm_video.mp4", fps=30):
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(self.frames, fps=fps)
        clip.write_videofile(filename)
        clip.close()
        self.frames = []
        matplotlib.pyplot.close(self.fig_data["figure"])
        self.fig_data = None