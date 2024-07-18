from GUI import *
import numpy as np
import cv2 as cv
import torch as t

class Image:
    def __init__(self, Data: np.array) -> None:
        if len(Data.shape) == 3:
            self.data = Data
            self.original_width = Data.shape[1]
            self.original_height = Data.shape[0]
            self.original_size = [self.original_width, self.original_height]
            self.aspect = self.original_height / self.original_width
            
            # Learning Variables
            self.scale_factor = 1
            self.position = [0, 0]
        else:
            raise Exception("Image Data Not in Right Shape")

def LoadImages(Path):
    # 打开图片文件
    img = cv.imread(Path)
    ImageIns = Image(img)
    return ImageIns
