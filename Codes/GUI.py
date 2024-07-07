import sys
import tkinter as tk

def get_canvas_size():
    # 创建一个Tk窗口对象
    window = tk.Tk()
    window.title("Canvas Size Input")
    
    window.geometry("400x300") 

    CanvasSize = {"width":1920,"height":1080,"margin":20}
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
    def on_submit():
        try:
            CanvasSize["width"] = int(entry_width.get())  # 将输入转换为整数
            CanvasSize["height"] = int(entry_height.get())  # 将输入转换为整数
            CanvasSize["margin"] = int(entry_margin.get())
            window.destroy()  # 关闭窗口
            window.quit()
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter valid numbers for width and height.")
            return None  # 如果输入无效，返回None

    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack()

    # 运行Tk主循环
    window.mainloop()

    return CanvasSize
