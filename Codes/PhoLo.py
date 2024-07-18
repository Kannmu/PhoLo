from Files import *
from GUI import *
from Optimize import *
import numpy as np


if __name__ == "__main__":
    # Get Images Paths
    ImagesPaths = SelectImages()

    ImagesList = []

    # Print Images Paths
    for i, path in enumerate(ImagesPaths):
        TempImageIns = LoadImages(path)
        ImagesList.append(TempImageIns)
        print(Fore.BLUE + 
            f"Loading images ({100*round((i+1)/(len(ImagesPaths)),2)}%\t{i+1}/{len(ImagesPaths)}): {path} \t Size: {TempImageIns.original_size}"
        )

    # Input Target Canvas Size
    CanvasSize = get_canvas_size()
    print(Fore.GREEN + "Target Canvas Size:", CanvasSize)

    CanvasIns = Canvas(CanvasSize)

    OptimizerIns = Optimizer(ImagesList, CanvasIns)

    OptimizerIns.Step()

