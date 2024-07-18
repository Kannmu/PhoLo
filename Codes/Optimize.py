import math
import numpy as np
import cv2 as cv
from Files import *
from PhoLo import Canvas
from GUI import *
from colorama import Fore
import torch as t

class Optimizer():
    def __init__(self,ImagesList: list[Image], CanvasSize: Canvas , LearningRate = 0.001) -> None:
        self.ImagesList = ImagesList

        self.CanvasSize = CanvasSize
        
        self.AllParams = t.tensor(t.zeros([len(ImagesList), 3]), dtype = t.float32, requires_grad = True)
        
        self.optimazer = t.optim.SGD(params=[self.AllParams], lr=LearningRate)
        
        print(self.AllParams.shape)

        self.GetStackCanvas()


    def Step(self):
        # Zero grad
        self.optimazer.zero_grad()

        # Calculate Loss
        self.Loss()

        # Backward flow loss
        self.LossValue.backward()

        # Update parameters
        self.optimazer.step()

    def CalPicPerPixel(self) -> float:
        return t.tensor(self.CanvasArray.sum() / self.CanvasArray.numel())

    def CalPixelsMoreThanOne(self) -> float:
        MoreThanOneArray = self.CanvasArray.clone().detach()
        MoreThanOneArray[MoreThanOneArray <= 1] = 0
        MoreThanOneArray[MoreThanOneArray > 1] = 1
        return t.tensor(MoreThanOneArray.sum())

    def GetStackCanvas(self):
        if len(self.ImagesList) == 0:
            return t.zeros(self.CanvasSize.marginal_size, dtype = t.float32, requires_grad=True)

        self.CanvasArray = t.zeros(self.CanvasSize.marginal_size, dtype = t.float32, requires_grad=True)

        for i, Params in enumerate(self.AllParams):
            scaled_width = t.tensor(int(self.ImagesList[i].original_width * Params[0]), dtype = t.float32)
            scaled_height = t.tensor(int(self.ImagesList[i].original_height * Params[0]), dtype = t.float32)

            top_left_x = t.max(t.tensor(0), Params[1])
            top_left_y = t.max(t.tensor(0), Params[2])
            bottom_right_x = t.min(t.tensor(self.CanvasSize.marginal_size[0]), top_left_x + scaled_width)
            bottom_right_y = t.min(t.tensor(self.CanvasSize.marginal_size[1]), top_left_y + scaled_height)

            # 如果图片在画布范围内
            if top_left_x <= bottom_right_x and top_left_y <= bottom_right_y:
                # 标记画布上被图片覆盖的像素点
                # 标记画布上被图片覆盖的像素点
                new_canvas = self.CanvasArray.clone()
                new_canvas[int(top_left_x):int(bottom_right_x), int(top_left_y):int(bottom_right_y)] += 1
                self.CanvasArray = new_canvas

        return self.CanvasArray

    def Loss(self):
        self.LossValue = t.tensor(0,dtype=t.float32, requires_grad=True)
        MaxX = t.tensor(len(self.ImagesList) + 1e-3, dtype = t.float32)

        # Mean Image Pixel Num Per Canvas Pixel
        PicPerPixel = self.CalPicPerPixel()

        if PicPerPixel <= 1:
            X = PicPerPixel
            PicPerPixelLoss = t.pow(((X * t.log(X) + 1) / X) - 1, 0.7)
        else:
            X = (MaxX - PicPerPixel) / (MaxX - 1)
            PicPerPixelLoss = t.pow(((X * t.log(X) + 1) / X) - 1, 0.7)

        self.LossValue = t.add(self.LossValue, PicPerPixelLoss.clone().detach().requires_grad_(True))


        # No overlap
        PixelsMoreThanOne = self.CalPixelsMoreThanOne()
        PixelsMoreThanOneLoss = t.log(PixelsMoreThanOne)
        self.LossValue = t.add(self.LossValue, PixelsMoreThanOneLoss.clone().detach().requires_grad_(True))

        # Print Loss Information
        print(Fore.RED+
            f"CalPicPerPixel:{PicPerPixel} \t Caused Loss: {PicPerPixelLoss} \t Percentage: {round((100*PicPerPixelLoss/self.LossValue ).item(), 3)}%"
        )
        print(Fore.RED+
            f"PixelsMoreThanOne: {PixelsMoreThanOne} \t Caused Loss: {PixelsMoreThanOneLoss}\t Percentage: {round((100*PixelsMoreThanOneLoss/self.LossValue ).item(), 3)}%"
        )
        
        print(Fore.YELLOW + str(len(self.ImagesList)), "Images, with initial Loss at: ", self.LossValue.item() )
        
