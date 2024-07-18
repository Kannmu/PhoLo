import sys
import tkinter as tk
from tkinter import filedialog

class Canvas:
    def __init__(self,CanvasSize:dict) -> None:
        self.original_size = [CanvasSize["width"],CanvasSize["height"]]
        self.margin = CanvasSize["margin"]
        self.marginal_size = [self.original_size[0] - 2*self.margin, self.original_size[1] - 2*self.margin]

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

    window.geometry("400x300")

    CanvasSize = {"width": 1920, "height": 1080, "margin": 20}
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
            window.destroy()  # 关闭窗口
            window.quit()
        except ValueError:
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
