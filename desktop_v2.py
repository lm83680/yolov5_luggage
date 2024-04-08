import tkinter as tk
from tkinter import ttk
import desktop_v2_detect  as v2_detect
import keyboard


person_height = 170

# 定义快捷键对应的操作函数
def on_key_press(event):
    if keyboard.is_pressed('ctrl') and keyboard.is_pressed('e'):
        end()

def start():
    v2_detect.v2_is_running = True
    print("程序已启动")
    v2_detect.run(person_height=person_height)

def end():
    v2_detect.v2_is_running = False
    print("程序已结束")

def on_select(event):
    global person_height
    person_height = int(combo.get())

root = tk.Tk()
root.title("超大行李智能识别v2.0")
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
root.geometry("300x200+%d+%d" %(ws/4,hs/4))
root.configure(background="#f0f0f0")
frame = ttk.Frame(root)

# 创建键盘监听器
keyboard.on_press(on_key_press)

# 创建启动按钮
start_button = ttk.Button(frame, text="启动", command=start, style='my.TButton')
start_button.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

# 创建结束按钮
end_button = ttk.Button(frame, text="Ctrl + E 结束程序", command=end, state=tk.DISABLED, style='my.TButton')
end_button.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)


# 创建标签
label = ttk.Label(root, text="参考人物身高（cm）：")
label.pack(pady=2)

# 创建下拉框
combo = ttk.Combobox(root, style='My.TCombobox', state='readonly')
combo['values'] = list(range(140, 201, 5))  # 设置可选值范围
combo.current(6)  # 默认值为 170，索引为 (170-140)/5=6
combo.bind('<<ComboboxSelected>>', on_select)
combo.pack(pady=10)

# 设置圆角
root.overrideredirect(True) # 消除默认边框
root.update_idletasks()
root.overrideredirect(False) # 设置自定义边框

# 设置按钮样式
style = ttk.Style()
style.configure('my.TButton', foreground='black', background='#4CAF50', font=('Helvetica', 12), borderwidth=2, relief="groove", padding=10)

# 创建下拉框样式
style_combo = ttk.Style()
# 设置下拉框样式
style_combo.configure('My.TCombobox', width=10)  # 设置宽度

# 显示框架
frame.pack(expand=True)

# 窗口循环
root.mainloop()
