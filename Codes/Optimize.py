import math
import os
import numpy as np
from Files import *
from GUI import *
from colorama import Fore
from Files import *

class Optimizer:
    def __init__(
        self, ImagesList, CanvasSize, InitialTemperature=10000.0, CoolingRate=0.99
    ) -> None:
        self.ImagesList = ImagesList
        self.CanvasSize = CanvasSize

        # Initialize all parameters
        self.ImageParams = np.ones((len(ImagesList), 3))
        self.CanvasArray = np.zeros(self.CanvasSize.original_size,dtype=np.int32)
        self.temperature = InitialTemperature
        self.cooling_rate = CoolingRate
        self.LayoutViewer = LayoutViewer(self.CanvasSize.original_size)

        if self.CanvasSize.auto_size:
            self.CanvasParams = np.zeros(np.asarray(self.CanvasSize.original_size).shape)
        else:
            self.CanvasParams = None

        self.LossList = []

    def ParticleStep(self, num_particles=30, inertia_weight=0.5, cognitive_coef=1.5, social_coef=1.5):
        # Step Function for simulating particle swarm optimization algorithm

        # Initialize particles if not already done
        if not hasattr(self, 'particles'):
            self.particles = {
                'positions': np.random.rand(num_particles, len(self.ImagesList), 3),
                'velocities': np.random.randn(num_particles, len(self.ImagesList), 3) * 0.1,
                'best_positions': np.random.rand(num_particles, len(self.ImagesList), 3),
                'best_scores': np.full(num_particles, np.inf)
            }
            if self.CanvasSize.auto_size:
                self.particles['positions_canvas'] = np.random.rand(num_particles, *self.CanvasParams.shape)
                self.particles['velocities_canvas'] = np.random.randn(num_particles, *self.CanvasParams.shape) * 0.1
                self.particles['best_positions_canvas'] = np.random.rand(num_particles, *self.CanvasParams.shape)
                self.particles['best_scores_canvas'] = np.full(num_particles, np.inf)

        # Evaluate loss for each particle
        for i in range(num_particles):
            self.ImageParams = self.particles['positions'][i]
            if self.CanvasSize.auto_size:
                self.CanvasParams = self.particles['positions_canvas'][i]

            current_loss, _ = self.Loss()

            # Update personal bests
            if current_loss < self.particles['best_scores'][i]:
                self.particles['best_scores'][i] = current_loss
                self.particles['best_positions'][i] = self.ImageParams.copy()
                if self.CanvasSize.auto_size:
                    self.particles['best_positions_canvas'][i] = self.CanvasParams.copy()

        # Find the global best
        global_best_index = np.argmin(self.particles['best_scores'])
        global_best_position = self.particles['best_positions'][global_best_index]
        if self.CanvasSize.auto_size:
            global_best_position_canvas = self.particles['best_positions_canvas'][global_best_index]

        # Update velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            self.particles['velocities'][i] = (
                inertia_weight * self.particles['velocities'][i]
                + cognitive_coef * r1 * (self.particles['best_positions'][i] - self.particles['positions'][i])
                + social_coef * r2 * (global_best_position - self.particles['positions'][i])
            )
            self.particles['positions'][i] += self.particles['velocities'][i]

            if self.CanvasSize.auto_size:
                r1, r2 = np.random.rand(2)
                self.particles['velocities_canvas'][i] = (
                    inertia_weight * self.particles['velocities_canvas'][i]
                    + cognitive_coef * r1 * (self.particles['best_positions_canvas'][i] - self.particles['positions_canvas'][i])
                    + social_coef * r2 * (global_best_position_canvas - self.particles['positions_canvas'][i])
                )
                self.particles['positions_canvas'][i] += self.particles['velocities_canvas'][i]

            # Clamp parameters to valid ranges
            self.particles['positions'][i][:, 0] = np.clip(self.particles['positions'][i][:, 0], 1 / len(self.ImagesList), 1)
            self.particles['positions'][i][:, 1:] = np.clip(self.particles['positions'][i][:, 1:], 0, 1)
            if self.CanvasSize.auto_size:
                self.particles['positions_canvas'][i] = np.clip(self.particles['positions_canvas'][i], 0, 1)

        # Set the best global position as the current parameters
        self.ImageParams = global_best_position
        if self.CanvasSize.auto_size:
            self.CanvasParams = global_best_position_canvas

        # Calculate new loss
        self.CurrentLoss, LossDetails = self.Loss()

        self.LossList.append(self.CurrentLoss)

        os.system("cls")

        print(Fore.RED + "Current Loss: ", self.CurrentLoss)
        print(Fore.YELLOW, [list(i) for i in self.ImageParams])

        print(
            Fore.GREEN
            + f"Pic Per Pixels: {round(LossDetails[0],3)} \t Caused Loss: {round(LossDetails[1],3)}\t Percentage: {round((100*LossDetails[1]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"OverLap Pixels : {round(LossDetails[2],3)} \t Caused Loss: {round(LossDetails[3],3)}\t Percentage: {round((100*LossDetails[3]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"Gap STDs: {round(LossDetails[4],3)} \t Caused Loss: {round(LossDetails[5],3)}\t Percentage: {round((100*LossDetails[5]/self.CurrentLoss ), 3)}%"
        )
        
        print(
            Fore.BLUE + str(len(self.ImagesList)),
            "Images, with current Loss at: ",
            self.CurrentLoss,
        )

        self.DrawRect(Block=False)

    def AdamStep(self, Lr=0.1, Beta1=0.9, Beta2=0.999, Epsilon=1e-8):
        # Step Function for simulating Adam algorithm
        
        # Initialize moment estimates if not already done
        if not hasattr(self, 'm'):
            self.m = {'image': np.zeros_like(self.ImageParams), 'canvas': np.zeros_like(self.CanvasParams) if self.CanvasSize.auto_size else None}
        if not hasattr(self, 'v'):
            self.v = {'image': np.zeros_like(self.ImageParams), 'canvas': np.zeros_like(self.CanvasParams) if self.CanvasSize.auto_size else None}
        if not hasattr(self, 't'):
            self.t = 0

        # Increment time step
        self.t += 1

        # Calculate current loss
        self.CurrentLoss, LossDetails = self.Loss()

        # Gradient calculation
        gradients = self.calculate_gradients()

        # Update biased first moment estimate
        self.m['image'] = Beta1 * self.m['image'] + (1 - Beta1) * gradients['image']
        if self.CanvasSize.auto_size:
            self.m['canvas'] = Beta1 * self.m['canvas'] + (1 - Beta1) * gradients['canvas']

        # Update biased second raw moment estimate
        self.v['image'] = Beta2 * self.v['image'] + (1 - Beta2) * (gradients['image'] ** 2)
        if self.CanvasSize.auto_size:
            self.v['canvas'] = Beta2 * self.v['canvas'] + (1 - Beta2) * (gradients['canvas'] ** 2)

        # Compute bias-corrected first moment estimate
        m_hat_image = self.m['image'] / (1 - Beta1 ** self.t)
        if self.CanvasSize.auto_size:
            m_hat_canvas = self.m['canvas'] / (1 - Beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat_image = self.v['image'] / (1 - Beta2 ** self.t)
        if self.CanvasSize.auto_size:
            v_hat_canvas = self.v['canvas'] / (1 - Beta2 ** self.t)

        # Update parameters
        self.ImageParams -= Lr * m_hat_image / (np.sqrt(v_hat_image) + Epsilon)
        if self.CanvasSize.auto_size:
            self.CanvasParams -= Lr * m_hat_canvas / (np.sqrt(v_hat_canvas) + Epsilon)

        # Clamp parameters to valid ranges
        self.ImageParams[:, 0] = np.clip(self.ImageParams[:, 0], 1 / len(self.ImagesList), 1)
        self.ImageParams[:, 1:] = np.clip(self.ImageParams[:, 1:], 0, 1)

        if self.CanvasSize.auto_size:
            self.CanvasParams = np.clip(self.CanvasParams, 0, 1)

        # Calculate new loss
        new_Loss, new_LossDetails = self.Loss()

        # Accept new solution if loss has decreased
        if new_Loss < self.CurrentLoss:
            self.CurrentLoss = new_Loss
            LossDetails = new_LossDetails
        else:
            # Revert to old solution if no improvement
            self.ImageParams += Lr * m_hat_image / (np.sqrt(v_hat_image) + Epsilon)
            if self.CanvasSize.auto_size:
                self.CanvasParams += Lr * m_hat_canvas / (np.sqrt(v_hat_canvas) + Epsilon)

        self.LossList.append(self.CurrentLoss)

        os.system("cls")

        print(Fore.RED + "Current Loss: ", self.CurrentLoss)
        print(Fore.YELLOW, [list(i) for i in self.ImageParams])

        print(
            Fore.GREEN
            + f"Pic Per Pixels: {round(LossDetails[0],3)} \t Caused Loss: {round(LossDetails[1],3)}\t Percentage: {round((100*LossDetails[1]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"OverLap Pixels : {round(LossDetails[2],3)} \t Caused Loss: {round(LossDetails[3],3)}\t Percentage: {round((100*LossDetails[3]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"Gap STDs: {round(LossDetails[4],3)} \t Caused Loss: {round(LossDetails[5],3)}\t Percentage: {round((100*LossDetails[5]/self.CurrentLoss ), 3)}%"
        )
        
        print(
            Fore.BLUE + str(len(self.ImagesList)),
            "Images, with current Loss at: ",
            self.CurrentLoss,
        )

        self.DrawRect(Block=False)
        self.LayoutViewer.draw_loss_curve(self.LossList)
        self.LayoutViewer.step()


    def SGDStep(self,Lr = 0.01):
        # Step Function for simulating stochastic gradient descent algorithm
        
        # Calculate current loss
        self.CurrentLoss, LossDetails = self.Loss()

        # Gradient calculation
        gradients = self.calculate_gradients()
        
        # Update parameters using gradients
        learning_rate = Lr  # Set a learning rate for SGD
        self.ImageParams -= learning_rate * gradients['image']
        if self.CanvasSize.auto_size:
            self.CanvasParams -= learning_rate * gradients['canvas']

        # Clamp parameters to valid ranges
        self.ImageParams[:, 0] = np.clip(self.ImageParams[:, 0], 1 / len(self.ImagesList), 1)
        self.ImageParams[:, 1:] = np.clip(self.ImageParams[:, 1:], 0, 1)

        if self.CanvasSize.auto_size:
            self.CanvasParams = np.clip(self.CanvasParams, 0, 1)

        # Calculate new loss
        new_Loss, new_LossDetails = self.Loss()

        # Accept new solution if loss has decreased
        if new_Loss < self.CurrentLoss:
            self.CurrentLoss = new_Loss
            LossDetails = new_LossDetails
        else:
            # Revert to old solution if no improvement
            self.ImageParams += learning_rate * gradients['image']
            if self.CanvasSize.auto_size:
                self.CanvasParams += learning_rate * gradients['canvas']

        self.LossList.append(self.CurrentLoss)

        os.system("cls")

        print(Fore.RED + "Current Loss: ", self.CurrentLoss)
        print(Fore.YELLOW, [list(i) for i in self.ImageParams])

        print(
            Fore.GREEN
            + f"Pic Per Pixels: {round(LossDetails[0],3)} \t Caused Loss: {round(LossDetails[1],3)}\t Percentage: {round((100*LossDetails[1]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"OverLap Pixels : {round(LossDetails[2],3)} \t Caused Loss: {round(LossDetails[3],3)}\t Percentage: {round((100*LossDetails[3]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"Gap STDs: {round(LossDetails[4],3)} \t Caused Loss: {round(LossDetails[5],3)}\t Percentage: {round((100*LossDetails[5]/self.CurrentLoss ), 3)}%"
        )
        print(
            Fore.BLUE + str(len(self.ImagesList)),
            "Images, with current Loss at: ",
            self.CurrentLoss,
        )

        self.DrawRect(Block=False)
    
    def calculate_gradients(self):
        # Placeholder function to calculate gradients
        # This function should return a dictionary with gradients for image and canvas parameters
        gradients = {
            'image': np.random.normal(size=self.ImageParams.shape),
            'canvas': np.random.normal(size=self.CanvasParams.shape) if self.CanvasSize.auto_size else np.zeros_like(self.CanvasParams)
        }
        return gradients


    def SAStep(self):
        # Step Function for simulating annealing algorithm

        # Calculate current loss
        self.CurrentLoss, LossDetails = self.Loss()

        TempParam = self.ImageParams.copy()
        if self.CanvasSize.auto_size:
            TempCanvasParam = self.CanvasParams.copy()

        # Generate new parameters using vectorized operations
        random_adjustments = 0.5 * np.random.normal(size=self.ImageParams.shape)
        new_Param = TempParam + random_adjustments
        new_Param[:, 0] = np.clip(new_Param[:, 0], 1 / len(self.ImagesList), 1)
        new_Param[:, 1:] = np.clip(new_Param[:, 1:], 0, 1)

        if self.CanvasSize.auto_size:
            new_CanvasParam = TempCanvasParam + 0.5 * np.random.normal(size=self.CanvasParams.shape)
            new_CanvasParam = np.clip(new_CanvasParam, 0, 1)
            self.CanvasParams = new_CanvasParam  # Temporarily switch to new solution

        self.ImageParams = new_Param  # Temporarily switch to new solution

        new_Loss, new_LossDetails = self.Loss()
        acceptance_probability = np.exp((self.CurrentLoss - new_Loss) / self.temperature)
        if new_Loss < self.CurrentLoss or acceptance_probability > np.random.rand():
            self.CurrentLoss = new_Loss
            LossDetails = new_LossDetails
            self.ImageParams = new_Param  # Accept new solution
            if self.CanvasSize.auto_size:
                self.CanvasParams = new_CanvasParam
        else:
            # Revert to old solution
            self.ImageParams = TempParam
            if self.CanvasSize.auto_size:
                self.CanvasParams = TempCanvasParam

        # Cool down
        self.temperature *= self.cooling_rate

        self.CurrentLoss.append(self.CurrentLoss)

        os.system("cls")

        print(Fore.RED + "Temperature: ", self.temperature)
        print(Fore.YELLOW, [list(i) for i in self.ImageParams])

        print(
            Fore.GREEN
            + f"Pic Per Pixels: {round(LossDetails[0],3)} \t Caused Loss: {round(LossDetails[1],3)}\t Percentage: {round((100*LossDetails[1]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"OverLap Pixels : {round(LossDetails[2],3)} \t Caused Loss: {round(LossDetails[3],3)}\t Percentage: {round((100*LossDetails[3]/self.CurrentLoss ), 3)}%"
        )

        print(
            Fore.GREEN
            + f"Gap STDs: {round(LossDetails[4],3)} \t Caused Loss: {round(LossDetails[5],3)}\t Percentage: {round((100*LossDetails[5]/self.CurrentLoss ), 3)}%"
        )
        print(
            Fore.BLUE + str(len(self.ImagesList)),
            "Images, with current Loss at: ",
            self.CurrentLoss,
        )

        self.DrawRect(Block=False)
    def ApplyParams(self):
        # Update Auto Canvas Params
        if self.CanvasSize.auto_size:
            self.CanvasSize.UpdateOptimalCanvasSize(((self.CanvasParams + 1) * self.CanvasSize.original_size).astype(int).tolist())
            self.LayoutViewer.CanvasSize = self.CanvasSize.optimal_size

        # Update Image Params
        for i, Img in enumerate(self.ImagesList):
            Img.UpdateCanvasSize(self.CanvasSize)
            Img.UpdatedPos(self.ImageParams[i])

    def DrawRect(self, Block:bool = False):
        self.RectList = []
        self.ApplyParams()
        for i, Img in enumerate(self.ImagesList):
            self.RectList.append(
                [
                    [
                        Img.top_left_x, 
                        Img.top_left_y
                    ],    
                    [
                        Img.bottom_right_x,
                        Img.bottom_right_y,
                    ],
                ]
            )
        
        self.LayoutViewer.draw_scaled_rectangles(self.RectList)

        if Block:
            input("Press Enter to continue...")

    def CalPicPerPixel(self):
        return self.CanvasArray.sum() / self.CanvasArray.size

    def CalPixelsMoreThanOne(self):
        MoreThanOneArray = self.CanvasArray.copy()
        MoreThanOneArray[MoreThanOneArray <= 1] = 0
        MoreThanOneArray[MoreThanOneArray > 1] = 1
        return MoreThanOneArray.sum()

    def CalPixelsOutOfCanvas(self):
        total_pixels = 0
        inside_pixels = 0

        canvas_width = self.CanvasSize.optimal_size[0] if self.CanvasSize.auto_size else self.CanvasSize.original_size[0]
        canvas_height = self.CanvasSize.optimal_size[1] if self.CanvasSize.auto_size else self.CanvasSize.original_size[1]
        for i, Params in enumerate(self.ImageParams):
            total_pixels += (
                self.ImagesList[i].scaled_width * self.ImagesList[i].scaled_height
            )
            # Calculate the coordinates that are within the canvas
            inside_top_left_x = max(0, self.ImagesList[i].top_left_x)
            inside_top_left_y = max(0, self.ImagesList[i].top_left_y)
            inside_bottom_right_x = min(
                canvas_width, self.ImagesList[i].bottom_right_x
            )
            inside_bottom_right_y = min(
                canvas_height, self.ImagesList[i].bottom_right_y
            )
            # Calculate the width and height of the inside rectangle
            inside_width = max(0, inside_bottom_right_x - inside_top_left_x)
            inside_height = max(0, inside_bottom_right_y - inside_top_left_y)
            inside_pixels += inside_width * inside_height

        pixels_out_of_canvas = total_pixels - inside_pixels
        return pixels_out_of_canvas

    def GetGapSTD(self, N:int):
        # Create a binary array where pixels with value >= 1 are set to 1, others to 0
        GapArray = (self.CanvasArray >= 1).astype(int)
        # Sample the CanvasArray every N pixels
        sampled_col_values = GapArray[::N, :]
        sampled_row_values = GapArray[:, ::N]
        
        # Calculate the standard deviation of consecutive zeros in each row
        row_consecutive_zeros_stds = self.CalConsecutiveZerosSTD(sampled_row_values.T)
        
        # Calculate the standard deviation of consecutive zeros in each column
        col_consecutive_zeros_stds = self.CalConsecutiveZerosSTD(sampled_col_values)

        # Compute the overall average standard deviation of consecutive zeros counts
        overall_avg_consecutive_zeros_std = np.std(row_consecutive_zeros_stds + col_consecutive_zeros_stds)

        return overall_avg_consecutive_zeros_std

    def CalConsecutiveZerosSTD(self, arr):
        consecutive_zeros_counts = []
        for row in arr:
            # Find where the row elements are zero
            is_zero = (row == 0)
            # Calculate the differences between consecutive elements in the zero array
            diff = np.diff(is_zero.astype(int))
            # Identify the start and end of consecutive zero sequences
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            # If the row starts with zeros, prepend a start index of 0
            if is_zero[0]:
                starts = np.insert(starts, 0, 0)
        
            # If the row ends with zeros, append an end index of the row length
            if is_zero[-1]:
                ends = np.append(ends, len(is_zero))
            # Calculate the lengths of consecutive zero sequences
            lengths = ends - starts
            # Calculate the standard deviation of these lengths and add to the list
            if len(lengths) > 0:
                for Count in lengths:
                    consecutive_zeros_counts.append(Count)
            else:
                pass
            
        # Calculate and return the sum of the standard deviations
        total_counts_list = consecutive_zeros_counts

        return total_counts_list
    
    def GetStackCanvas(self):
        # Initialize the canvas array with zeros
        canvas_size = self.CanvasSize.optimal_size if self.CanvasSize.auto_size else self.CanvasSize.original_size
        
        self.CanvasArray = np.zeros(canvas_size, dtype=np.int32)
        
        # Apply the parameters to adjust the canvas and image positions
        self.ApplyParams()

        # Iterate over each image and update the canvas array
        for img in self.ImagesList:
            top_left_x = max(0, img.top_left_x)
            top_left_y = max(0, img.top_left_y)

            bottom_right_x = min(canvas_size[0], img.top_left_x + img.scaled_width)
            bottom_right_y = min(canvas_size[1], img.top_left_y + img.scaled_height)
            if top_left_x < bottom_right_x and top_left_y < bottom_right_y:
                self.CanvasArray[top_left_x:bottom_right_x, top_left_y:bottom_right_y] += 1

    def Loss(self):
        # Get Newest Canvas Array
        self.GetStackCanvas()

        MaxX = len(self.ImagesList) + 1e-5

        # Mean Image Pixel Num Per Canvas Pixel
        PicPerPixel = self.CalPicPerPixel()

        # CalCulate Pic Per Pixel Loss
        if PicPerPixel <= 1:
            X = PicPerPixel
            PicPerPixelLoss = 5 * math.pow(((X * np.log(X) + 1) / X) - 1, 0.7)
        else:
            X = (MaxX - PicPerPixel) / (MaxX - 1)
            PicPerPixelLoss = 5 * math.pow(((X * np.log(X) + 1) / X) - 1, 0.7)

        # No overlap
        PixelsMoreThanOne = self.CalPixelsMoreThanOne()
        PixelsMoreThanOneLoss = 10 * np.log(PixelsMoreThanOne + 1)

        # No Out of Canvas
        # PixelsOutOfCanvas = self.CalPixelsOutOfCanvas()
        # PixelsOutOfCanvasLoss = np.log(PixelsOutOfCanvas + 1)

        # Blank STDs
        if self.CanvasSize.auto_size:
            GapSTD = self.GetGapSTD(int(0.1 * self.CanvasSize.optimal_size[0]))
        else:
            GapSTD = self.GetGapSTD(int(0.1 * self.CanvasSize.original_size[0]))
        GapLoss = 10*np.log(GapSTD + 1)

        # Accumulate Loss
        Loss = PicPerPixelLoss + PixelsMoreThanOneLoss + GapLoss + 1e-5

        return Loss, [
            PicPerPixel,
            PicPerPixelLoss,
            PixelsMoreThanOne,
            PixelsMoreThanOneLoss,
            # PixelsOutOfCanvas,
            # PixelsOutOfCanvasLoss,
            GapSTD,
            GapLoss
        ]

