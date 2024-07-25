from GUI import *
import numpy as np
import cv2 as cv

class Image:
    def __init__(self, Data: np.array, CanvasSize) -> None:
        if len(Data.shape) == 3:
            self.data = Data
            self.CanvasSize = CanvasSize
            self.original_width = Data.shape[1]
            self.original_height = Data.shape[0]
            self.original_size = [self.original_width, self.original_height]
            self.aspect = self.original_height / self.original_width

            self.UpdateCanvasSize(CanvasSize)
            # self.CanvasAspect = self.CanvasSize.original_size[1]/self.CanvasSize.original_size[0]
            # self.PreScaleFactor = self.CanvasSize.original_size[1]/self.original_height if(self.aspect >= self.CanvasAspect) else self.CanvasSize.original_size[0]/self.original_width

            # Learning Variables
            self.scale_factor = 1
            self.position = [0, 0]
        else:
            raise Exception("Image Data Not in Right Shape")
    
    def UpdateCanvasSize(self, CanvasSize):
        self.CanvasSize = CanvasSize
        self.CanvasAspect = self.CanvasSize.optimal_size[1]/self.CanvasSize.optimal_size[0]
        self.PreScaleFactor = self.CanvasSize.optimal_size[1]/self.original_height if(self.aspect >= self.CanvasAspect) else self.CanvasSize.optimal_size[0]/self.original_width

    def UpdatedPos(self, Params):
        self.scaled_width = int(self.original_width * Params[0] * self.PreScaleFactor)
        self.scaled_height = int(self.original_height * Params[0] * self.PreScaleFactor)
        if(self.CanvasSize.auto_size):
            self.top_left_x = int(Params[1]*(self.CanvasSize.optimal_size[0]-self.scaled_width))
            self.top_left_y = int(Params[2]*(self.CanvasSize.optimal_size[1]-self.scaled_height))
        else:
            self.top_left_x = int(Params[1]*(self.CanvasSize.original_size[0]-self.scaled_width))
            self.top_left_y = int(Params[2]*(self.CanvasSize.original_size[1]-self.scaled_height))
        self.bottom_right_x = self.top_left_x + self.scaled_width
        self.bottom_right_y = self.top_left_y + self.scaled_height

def LoadImages(Path,CanvasIns:Canvas):
    # 打开图片文件
    img = cv.imread(Path)
    ImageIns = Image(img, CanvasIns)
    return ImageIns
