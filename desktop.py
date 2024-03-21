import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import detect_desktop as detect
import detect as detect_c


def get_folder_path():
    folder_path = filedialog.askdirectory().replace("/", "\\")
    entry_var.set(folder_path)
    result_var.set('')
    result_btn_open_path.config(state=tk.DISABLED, bg='lightgray')
    if entry_var.get():
        start_but.config(state=tk.NORMAL,bg='lightgreen')
        label_tip.config(text="程序状态：准备就绪")
    else:
        start_but.config(state=tk.DISABLED,bg='lightgray')
        label_tip.config(text="程序状态：请选择文件夹")


def start():
    label_tip.config(text="程序状态：正在识别中...")
    root.update()
    try:
        result_folder_path = detect.run(source=entry_var.get())
        if result_folder_path:
            result_var.set(f'{os.path.dirname(os.path.realpath(__file__))}\\{result_folder_path}')
            result_btn_open_path.config(state=tk.NORMAL, bg='lightblue')
            open_result_path()
            label_tip.config(text="程序状态：识别成功")
        else:
            result_btn_open_path.config(state=tk.DISABLED, bg='lightgray')
            label_tip.config(text="程序状态：识别失败")
    except Exception as e:
        label_tip.config(text=f"程序状态：{e}")


def start_stream():
    if label_tip.cget("text") != '程序状态：正在识别中...':
        detect_c.run(source='0')
        label_tip.config(text="程序状态：正在识别中...")
        root.update()


def open_result_path():
    try:
        subprocess.Popen(["explorer", os.path.realpath(result_var.get())])
    except Exception as e:
        print(f"无法打开目录: {e},路径是：{result_var.get()}")


# 创建主窗口
root = tk.Tk()
root.title("超大行李智能识别v1.0")
# 添加标签
label = tk.Label(root, text="",height=4)
label.grid(row=0, column=0)
# 添加标签
label = tk.Label(root, text="所在文件夹路径：")
label.grid(row=0, column=0)

entry_var = tk.StringVar()
entry_path = tk.Entry(root, textvariable=entry_var, state="readonly", width=67)
entry_path.grid(row=0, column=1)

# 添加获取文件夹路径的按钮和显示路径的 Entry
btn_get_path = tk.Button(root, text="选择文件夹", command=get_folder_path)
btn_get_path.grid(row=0, column=2)


# 添加提示
label_tip = tk.Label(root, text="程序状态：请选择文件夹")
label_tip.grid(row=3, column=0, columnspan=2)
start_but = tk.Button(root, text="开始识别", state=tk.DISABLED, command=start, bg="lightgray", width=10)
start_but.grid(row=1, column=2)

# 添加标签
result_label = tk.Label(root, text="", height=4)
result_label.grid(row=2, column=0)
# 添加标签
result_label = tk.Label(root, text="识别结果路径：")
result_label.grid(row=2, column=0)

result_var = tk.StringVar()
result_path = tk.Entry(root, textvariable=result_var, state="readonly", width=67)
result_path.grid(row=2, column=1)

# 添加获取文件夹路径的按钮和显示路径的 Entry
result_btn_open_path = tk.Button(root, text="打开文件夹", command=open_result_path, state=tk.DISABLED, bg="lightgray", width=10)
result_btn_open_path.grid(row=2, column=2)


# 开始 Tkinter 事件循环
root.mainloop()
