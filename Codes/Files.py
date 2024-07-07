import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2 as cv

class Image:
    def __init__(self,Data:np.array) -> None:
        if(len(Data.shape) == 3): 
            self.data = Data
            self.original_width = Data.shape[1]
            self.original_height = Data.shape[0]
            self.original_size = [self.original_width, self.original_height]
            self.aspect = self.original_height/self.original_width
            self.scale_factor = 1
            self.position = [0, 0]
        else:
            raise Exception("Image Data Not in Right Shape")
    



def LoadImages(Path):
    # 打开图片文件
    img = cv.imread(Path)  # 替换为你的图片路径

    # 打印数组的形状来确认
    # print(Path,"has shape of: ",img.shape)

    ImageIns = Image(img)

    return ImageIns

    
def SelectImages():
    # 创建一个Tk窗口对象，但不显示它
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 定义文件选择窗口的文件类型，同时包含.jpg和.png
    file_types = [
        ("Image files", ("*.jpg", "*.png")),  # 同时选择.jpg和.png文件
        ("All Files", "*.*")  # 允许选择所有文件
    ]

    # 打开文件选择窗口，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title='Select images',
        initialdir='/',  # 初始目录，可以按需修改
        filetypes=file_types,
        multiple=True  # 允许选择多个文件
    )
    if file_paths:
        # 返回选中的文件路径列表
        return file_paths
    else:
        return None

