import sys
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

class Canvas:
    def __init__(self,CanvasSize:dict) -> None:
        self.original_size = [CanvasSize["width"],CanvasSize["height"]]
        self.optimal_size = self.original_size
        self.auto_size = CanvasSize["AutoSize"]
        self.margin = CanvasSize["margin"]
        self.marginal_size = [self.original_size[0] - 2*self.margin, self.original_size[1] - 2*self.margin]

    def UpdateOptimalCanvasSize(self,CanvasSize) -> None:
        if(not self.auto_size):
            return None
        self.optimal_size = CanvasSize


def SelectImages():
    # 创建一个Tk窗口对象，但不显示它
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 定义文件选择窗口的文件类型，同时包含.jpg和.png
    file_types = [
        ("Image files", ("*.jpg", "*.png")),  # 同时选择.jpg和.png文件
        ("All Files", "*.*"),  # 允许选择所有文件
    ]

    # 打开文件选择窗口，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title="Select images",
        initialdir="/",  # 初始目录，可以按需修改
        filetypes=file_types,
        multiple=True,  # 允许选择多个文件
    )
    if file_paths:
        # 返回选中的文件路径列表
        return file_paths
    else:
        return None


def get_canvas_size():
    # 创建一个Tk窗口对象
    window = tk.Tk()
    window.title("Canvas Size Input")

    window.geometry("400x350")

    CanvasSize = {"width": 1920, "height": 1080, "margin": 20, "AutoSize": False}
    # 创建一个标签和输入框用于输入宽度
    label_width = tk.Label(window, text="Enter width:")
    label_width.pack()
    entry_width = tk.Entry(window)
    entry_width.pack()

    # 创建一个标签和输入框用于输入高度
    label_height = tk.Label(window, text="Enter height:")
    label_height.pack()
    entry_height = tk.Entry(window)
    entry_height.pack()

    # 创建一个标签和输入框用于输入高度
    label_margin = tk.Label(window, text="Enter margin:")
    label_margin.pack()
    entry_margin = tk.Entry(window)
    entry_margin.pack()

    # 创建一个选择框用于选择AutoSize
    auto_size_var = tk.BooleanVar()
    checkbox_auto_size = tk.Checkbutton(window, text="Auto Size", variable=auto_size_var)
    checkbox_auto_size.pack()
    def on_close():
        sys.exit(0)

    # 绑定关闭窗口的行为
    window.protocol("WM_DELETE_WINDOW", on_close)

    # 创建一个按钮，点击后获取输入并关闭窗口
    def on_submit(Fake = None):
        try:
            CanvasSize["width"] = int(entry_width.get())  # 将输入转换为整数
            CanvasSize["height"] = int(entry_height.get())  # 将输入转换为整数
            CanvasSize["margin"] = int(entry_margin.get())
            CanvasSize["AutoSize"] = auto_size_var.get()  # 获取选择框的值
            window.destroy()  # 关闭窗口
            window.quit()
        except ValueError:
            CanvasSize["AutoSize"] = auto_size_var.get()
            window.destroy()  # 关闭窗口
            window.quit()
            return CanvasSize  # 如果输入无效，返回Default

    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack()

    entry_width.bind("<Return>", on_submit)
    entry_height.bind("<Return>", on_submit)
    entry_margin.bind("<Return>", on_submit)

    # 运行Tk主循环
    window.mainloop()

    return CanvasSize

class LayoutViewer():
    def __init__(self, CanvasSize) -> None:
        plt.ion()
        self.CanvasSize = CanvasSize
        self.fig, (self.ax, self.ax_loss) = plt.subplots(1, 2, figsize=(15,6))
        # Create a new figure and axis
        
        self.ax.set_xlim(-50, self.CanvasSize[0] + 50)
        self.ax.set_ylim(-50, self.CanvasSize[1] + 50)

        
    def draw_scaled_rectangles(self, RectList):
        # 1. Clear previous drawing results
        self.ax.clear()

        self.colors = plt.cm.viridis(np.linspace(0, 1, len(RectList)))
        # 2. Draw the canvas border with a blue rectangle
        canvas_border = plt.Rectangle((0, 0), self.CanvasSize[0], self.CanvasSize[1], edgecolor='blue', fill=False, linewidth=4)
        self.ax.add_patch(canvas_border)
        plt.text(self.CanvasSize[0] / 2, 1.2*self.CanvasSize[1],'Canvas', fontsize=15, ha='center', va='center')
        # 3. Draw N rectangles from RectList with different colors
        for i, rect in enumerate(RectList):
            rectangle = plt.Rectangle(rect[0], rect[1][0] - rect[0][0], rect[1][1] - rect[0][1], edgecolor=self.colors[i], fill=False, linewidth=2)
            self.ax.add_patch(rectangle)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('equal')

    def draw_loss_curve(self, loss_list):
        # Clear the previous plot
        self.ax_loss.clear()
        # Plot the loss list on the new subplot
        self.ax_loss.plot(loss_list, linewidth=2)
        
        # Set titles and labels
        self.ax_loss.set_title('Loss Curve')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')

    def step(self):
        plt.pause(0.001)

