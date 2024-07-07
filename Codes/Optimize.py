import math
import numpy as np
import cv2 as cv
from Files import *
from PhoLo import Canvas

def CalPicPerPixel(ImagesList:list[Image],CanvasSize:Canvas)->float:
    if len(ImagesList) == 0:
        return 0.0

    CanvasArray = np.zeros(CanvasSize.marginal_size)

    for i, ImageIns in enumerate(ImagesList):
        scaled_width = int(ImageIns.original_width * ImageIns.scale_factor)
        scaled_height = int(ImageIns.original_height * ImageIns.scale_factor)

        top_left_x = max(0, ImageIns.position[0])
        top_left_y = max(0, ImageIns.position[1])
        bottom_right_x = min(CanvasSize.marginal_size[0], top_left_x + scaled_width)
        bottom_right_y = min(CanvasSize.marginal_size[1], top_left_y + scaled_height)

        # 如果图片在画布范围内
        if top_left_x <= bottom_right_x and top_left_y <= bottom_right_y:
            # 标记画布上被图片覆盖的像素点
            CanvasArray[top_left_x:bottom_right_x, top_left_y:bottom_right_y] += 1

    return CanvasArray.sum() / CanvasArray.size


def Loss(ImagesList:list[Image],CanvasSize:Canvas):
    X = CalPicPerPixel(ImagesList, CanvasSize)
    MaxX = len(ImagesList)
    if(X<=1):
        Loss = math.pow(((X*np.log(X)+1)/X)-1, 0.7)
    else:
        X = (MaxX-X)/(MaxX-1)
        Loss = math.pow(((X*np.log(X)+1)/X)-1, 0.7)
    return Loss


