"""
修改一个目录下所有的文件名为0001、0002...排序
"""
import os

def rename_files(directory):

    # 从编号0001开始
    start = 40
    # 获取目录下所有文件名
    files = os.listdir(directory)

    # 过滤出文件，而非文件夹
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

    # 对文件名进行排序
    files.sort()

    # 重命名文件
    for index, filename in enumerate(files, start=start):
        _, ext = os.path.splitext(filename)
        new_filename = f"{index:04d}{ext}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    # 指定目录路径
    # target_directory = "D:/Project/Python/yolov5/data/target"
    target_directory = "C:/Users/qinchangteng/Downloads/temp"

    # 调用函数进行重命名
    rename_files(target_directory)

    print("Files renamed successfully.")
