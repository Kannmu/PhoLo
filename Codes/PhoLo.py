import tqdm
from Files import *
from GUI import *
from Optimize import *

if __name__ == "__main__":
    
    # Input Target Canvas Size
    CanvasSize = get_canvas_size()
    print(Fore.GREEN + "Target Canvas Size:", CanvasSize)
    CanvasIns = Canvas(CanvasSize)

    # Get Images Paths
    ImagesPaths = SelectImages()

    ImagesList = []

    # Print Images Paths
    for i, path in enumerate(ImagesPaths):
        TempImageIns = LoadImages(path, CanvasIns)
        ImagesList.append(TempImageIns)
        print(Fore.BLUE + 
            f"Loading images ({100*round((i+1)/(len(ImagesPaths)),2)}%\t{i+1}/{len(ImagesPaths)}): {path} \t Size: {TempImageIns.original_size}"
        )

    OptimizerIns = Optimizer(ImagesList, CanvasIns)

    # OptimizerIns.CalConsecutiveZerosSTD(np.array([[1,1,1,0,2,2,2,2,2,0,0,2,22,2,2,2,2,0,0,0,22323,23,2,1,0,]]))

    for _ in tqdm.tqdm(range(1000)):
        # OptimizerIns.SAStep()
        # OptimizerIns.SGDStep(0.1)
        OptimizerIns.AdamStep(1)
        # OptimizerIns.ParticleStep()
    plt.ioff()
    plt.show()