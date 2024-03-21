# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
服务于win桌面应用的识别服务程序
"""

import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()  # 用于自动切换模型的推理模式，如果是FP16模型，则自动切换为FP16推理模式，否则切换为FP32推理模式，这样可以避免模型推理时出现类型不匹配的错误
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=True,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=6,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #################################################################初始化参数#################################################################
    source = str(source)  # 将source转换为字符串，source为输入的图片、视频、摄像头等
    save_img = not nosave and not source.endswith('.txt')  # 判断是否保存图片，如果nosave为False，且source不是txt文件，则保存图片
    is_file = Path(source).suffix[1:] in (
            IMG_FORMATS + VID_FORMATS)  # 判断source是否是文件.Path(source)使用source创建一个Path对象，用于获取输入源信息，suffix获取文件扩展名：.jpg,.mp4等，suffix[1:]获取文件后缀，判断后缀是否在IMG_FORMATS和VID_FORMATS中，如果是，则is_file为True
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://',
                                        'https://'))  # 判断source是否是url，如果是，则is_url为True.lower()将字符串转换为小写,startswith()判断字符串是否以指定的字符串开头
    webcam = source.isnumeric() or source.endswith('.streams') or (
            is_url and not is_file)  # source.isnumeric()判断source是否是数字，source.endswith('.streams')判断source是否以.streams结尾，(is_url and not is_file)判断source是否是url，且不是文件，上述三个条件有一个为True，则webcam为True。
    screenshot = source.lower().startswith('screen')  # 判断source是否是截图，如果是，则screenshot为True
    if is_url and is_file:
        source = check_file(source)  # 确保输入源为本地文件，如果是url，则下载到本地，check_file()函数用于下载url文件

    # Directories，创建保存结果的文件夹
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run，增加文件或目录路径，即运行/exp——>运行/exp{sep}2，运行/exp{sep}3，…等。exist_ok为True时，如果文件夹已存在，则不会报错
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir，创建文件夹，如果save_txt为True，则创建labels文件夹，否则创建save_dir文件夹

    # Load model,初始化模型
    device = select_device(device)  # 选择设备，如果device为空，则自动选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data,
                               fp16=half)  # 加载模型，DetectMultiBackend()函数用于加载模型，weights为模型路径，device为设备，dnn为是否使用opencv dnn，data为数据集，fp16为是否使用fp16推理
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型的stride，names，pt,model.stride为模型的stride，model.names为模型的类别，model.pt为模型的类型
    imgsz = check_img_size(imgsz, s=stride)  # check image size,验证图像大小是每个维度的stride=32的倍数

    # Dataloader,初始化数据集
    bs = 1  # batch_size,初始化batch_size为1
    if webcam:  # 如果source是摄像头，则创建LoadStreams()对象
        view_img = check_imshow(warn=True)  # 是否显示图片，如果view_img为True，则显示图片
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt,
                              vid_stride=vid_stride)  # 创建LoadStreams()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        bs = len(dataset)  # batch_size为数据集的长度
    elif screenshot:  # 如果source是截图，则创建LoadScreenshots()对象
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride,
                                  auto=pt)  # 创建LoadScreenshots()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,
                             vid_stride=vid_stride)  # 创建LoadImages()对象，直接加载图片，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化vid_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象

    #################################################################开始推理#################################################################
    # Run inference，运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3,
                        *imgsz))  # warmup，预热，用于提前加载模型，加快推理速度，imgsz为图像大小，如果pt为True或者model.triton为True，则bs=1，否则bs为数据集的长度。3为通道数，*imgsz为图像大小，即(1,3,640,640)
    seen, windows, dt = 0, [], (
        Profile(), Profile(), Profile())  # 初始化seen，windows，dt，seen为已检测的图片数量，windows为空列表，dt为时间统计对象
    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集，path为图片路径，im为图片，im0s为原始图片，vid_cap为视频读取对象，s为视频帧率
        with dt[0]:  # 开始计时,读取图片
            im = torch.from_numpy(im).to(model.device)  # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32    #如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
            im /= 255  # 0 - 255 to 0.0 - 1.0                         #将图片归一化，将图片像素值从0-255转换为0-1
            if len(im.shape) == 3:  # 如果图片的维度为3，则添加batch维度
                im = im[
                    None]  # expand for batch dim                 #在前面添加batch维度，即将图片的维度从3维转换为4维，即(3,640,640)转换为(1,3,640,640)，pytorch模型的输入必须是4维的

        # Inference
        with dt[1]:  # 开始计时,推理时间
            visualize = increment_path(save_dir / Path(path).stem,
                                       mkdir=True) if visualize else False  # 如果visualize为True，则创建visualize文件夹，否则为False
            pred = model(im, augment=augment,
                         visualize=visualize)  # 推理，model()函数用于推理，im为输入图片，augment为是否使用数据增强，visualize为是否可视化,输出pred为一个列表，形状为（n,6）,n代表预测框的数量，6代表预测框的坐标和置信度，类别

        # NMS，非极大值抑制，用于去除重复的预测框
        with dt[2]:  # 开始计时,NMS时间
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                       max_det=max_det)  # NMS，non_max_suppression()函数用于NMS，pred为输入的预测框，conf_thres为置信度阈值，iou_thres为iou阈值，classes为类别，agnostic_nms为是否使用类别无关的NMS，max_det为最大检测框数量，

        # 直到这一行，上面👆属于加载和推理过程，下面👇均属于事物处理过程

        # Process predictions,处理预测结果
        for i, det in enumerate(pred):  # per image,遍历每张图片,enumerate()函数将pred转换为索引和值的形式，i为索引，det为对应的元素，即每个物体的预测框
            seen += 1  # 检测的图片数量加1
            if webcam:  # batch_size >= 1，如果是摄像头，则获取视频帧率
                p, im0, frame = path[i], im0s[
                    i].copy(), dataset.count  # path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧数
                s += f'{i}: '  # 在打印输出中添加当前处理的图像索引号i，方便调试和查看结果。在此处，如果是摄像头模式，i表示当前批次中第i张图像；否则，i始终为0，因为处理的只有一张图像。
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # 如果不是摄像头，frame为0

            p = Path(p)  # to Path                             #将路径转换为Path对象
            save_path = str(save_dir / p.name)  # im.jpg，保存图片的路径，save_dir为保存图片的文件夹，p.name为图片名称
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt，保存预测框的路径，save_dir为保存图片的文件夹，p.stem为图片名称，dataset.mode为数据集的模式，如果是image，则为图片，否则为视频
            s += '%gx%g ' % im.shape[2:]  # print string,打印输出，im.shape[2:]为图片的宽和高
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh,归一化因子，用于将预测框的坐标从归一化坐标转换为原始坐标
            imc = im0.copy() if save_crop else im0  # for save_crop,如果save_crop为True，则将im0复制一份，否则为im0
            annotator = Annotator(im0, line_width=line_thickness,
                                  example=str(names))  # 创建Annotator对象，用于在图片上绘制预测框和标签,im0为输入图片，line_width为线宽，example为标签


            if len(det):  # 如果预测框的数量大于0

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                         im0.shape).round()  # 将预测框的坐标从归一化坐标转换为原始坐标,im.shape[2:]为图片的宽和高，det[:, :4]为预测框的坐标，im0.shape为图片的宽和高

                # Print results,打印输出
                for c in det[:, 5].unique():  # 遍历每个类别,unique()用于获取检测结果中不同类别是数量
                    n = (det[:, 5] == c).sum()  # detections per class                        #n为每个类别的预测框的数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string             #s为每个类别的预测框的数量和类别

                # Write results，写入结果
                the_pic_data = []  # 准备属于这张图像的List容器
                for *xyxy, conf, cls in reversed(det):  # 遍历每个预测框,xyxy为预测框的坐标，conf为置信度，cls为类别,reversed()函数用于将列表反转，*是一个扩展语法，*xyxy表示将xyxy中的元素分别赋值给x1,y1,x2,y2
                    if save_txt:  # Write to file,如果save_txt为True，则将预测框的坐标和类别写入txt文件中
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh,将预测框的坐标从原始坐标转换为归一化坐标
                        line = (cls, *xywh, conf) if save_conf else (
                            cls, *xywh)  # label format,如果save_conf为True，则将置信度也写入txt文件中
                        with open(f'{txt_path}.txt', 'a') as f:  # 打开txt文件,'a'表示追加
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # 写入txt文件

                    if save_img or save_crop or view_img:  # Add bbox to image,如果save_img为True，则将预测框和标签绘制在图片上
                        c = int(cls)  # integer class,获取类别
                        label = None if hide_labels else (names[
                                                              c] if hide_conf else f'{names[c]} {conf:.2f}')  # 如果hide_labels为True，则不显示标签，否则显示标签，如果hide_conf为True，则不显示置信度，否则显示置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制预测框和标签




            # Stream results,在图片上绘制预测框和标签展示
            im0 = annotator.result()  # 获取绘制预测框和标签的图片
            if view_img:  # 如果view_img为True，则展示图片
                if platform.system() == 'Linux' and p not in windows:  # 如果系统为Linux，且p不在windows中
                    windows.append(p)  # 将p添加到windows中
                    cv2.namedWindow(str(p),
                                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux),允许窗口调整大小,WINDOW_NORMAL表示用户可以调整窗口大小，WINDOW_KEEPRATIO表示窗口大小不变
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小，使其与图片大小一致
                cv2.imshow(str(p), im0)  # 显示图片
                cv2.waitKey(1)  # 1 millisecond                                        #等待1毫秒

            # Save results (image with detections)
            if save_img:  # 如果save_img为True，则保存图片
                if dataset.mode == 'image':  # 如果数据集模式为image
                    cv2.imwrite(save_path, im0)  # 保存图片
                else:  # 'video' or 'stream'，如果数据集模式为video或stream
                    if vid_path[i] != save_path:  # new video，如果vid_path[i]不等于save_path
                        vid_path[i] = save_path  # 将save_path赋值给vid_path[i]
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果vid_writer[i]是cv2.VideoWriter类型
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # 如果save_txt为True，则打印保存的标签数量
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 打印保存的路径
    if update:
        strip_optimizer(
            weights[0])  # update model (to fix SourceChangeWarning)                                      #更新模型


"""
    weights: 用于检测的模型路径
    source: 检测的路径，可以是图片，视频，文件夹，也可以是摄像头（‘0’）
    data: 数据集的配置文件，用于获取类别名称，和训练时的一样
    imgsz: 网络输入的图片大小，默认为640
    conf-thres: 置信度阈值，大于该阈值的框才会被保留
    iou-thres: NMS的阈值，大于该阈值的框会被合并，小于该阈值的框会被保留，一般设置为0.45
    max-det: 每张图片最多检测的目标数，默认为1000
    device: 检测的设备，可以是cpu，也可以是gpu，可以不用设置，会自动选择
    view-img: 是否显示检测结果，默认为False
    save-txt: 是否将检测结果保存为txt文件，包括类别，框的坐标，默认为False
    save-conf: 是否将检测结果保存为txt文件，包括类别，框的坐标，置信度，默认为False
    save-crop: 是否保存裁剪预测框的图片，默认为False
    nosave: 不保存检测结果，默认为False
    classes: 检测的类别，默认为None，即检测所有类别，如果设置了该参数，则只检测该参数指定的类别
    agnostic-nms: 进行NMS去除不同类别之间的框，默认为False
    augment: 推理时是否进行TTA数据增强，默认为False
    update: 是否更新模型，默认为False,如果设置为True，则会更新模型,对模型进行剪枝，去除不必要的参数
    project: 检测结果保存的文件夹，默认为runs/detect
    name: 检测结果保存的文件夹，默认为exp
    exist-ok: 如果检测结果保存的文件夹已经存在，是否覆盖，默认为False
    line-thickness: 框的线宽，默认为3
    hide-labels: 是否隐藏类别，默认为False
    hide-conf: 是否隐藏置信度，默认为False
    half: 是否使用半精度推理，默认为False
    dnn: 是否使用OpenCV的DNN模块进行推理，默认为False
    vid-stride: 视频帧采样间隔，默认为1，即每一帧都进行检测
"""


def main(opt):  # 主函数  作为包调用时，调用 main({'weights':'xxx','source':'xxx'})
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查依赖，如果没有安装依赖，则会自动安装
    run(**vars(opt))  # 运行程序，vars()函数返回对象object的属性和属性值的字典对象
