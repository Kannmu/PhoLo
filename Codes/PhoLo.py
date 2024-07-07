from Files import *
from GUI import *
from Optimize import *
import numpy as np

class Canvas:
    def __init__(self,CanvasSize:dict) -> None:
        self.original_size = [CanvasSize["width"],CanvasSize["height"]]
        self.margin = CanvasSize["margin"]
        self.marginal_size = [self.original_size[0] - 2*self.margin, self.original_size[1] - 2*self.margin]


if __name__ == "__main__":
    # Get Images Paths
    ImagesPaths = SelectImages()

    ImagesList = []

    # Print Images Paths
    for i,path in enumerate(ImagesPaths):
        TempImageIns = LoadImages(path)
        ImagesList.append(TempImageIns)
        print(f"Loading images ({100*round((i+1)/(len(ImagesPaths)),2)}%\t{i+1}/{len(ImagesPaths)}): {path} \t Size: {TempImageIns.original_size}")

    # Input Target Canvas Size
    CanvasSize = get_canvas_size()
    print("Target Canvas Size:", CanvasSize)

    CanvasIns = Canvas(CanvasSize)

    Loss = Loss(ImagesList, CanvasIns)
    
    print(len(ImagesList),"Images, with initial Loss at: ",Loss)



