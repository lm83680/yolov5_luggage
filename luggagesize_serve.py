# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
change from detect.py

主要目的实现获取目标的识别结果，读取目标参数，
"""

import argparse
import csv
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

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / "weight/date121size8epochs285.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    if is_url and is_file:
        source = check_file(source)  # download
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # 准备本次识别的结果容器
    result_data = []
    # result_data = [
    #     [
    #         {
    #             'label_name': "",
    #             'xywh': ""
    #         }
    #     ]
    # ]

    # 这个for 抽取了一张图像
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        with dt[1]:  # 开始计时,推理时间
            visualize = increment_path(save_dir / Path(path).stem,
                                       mkdir=True) if visualize else False  # 如果visualize为True，则创建visualize文件夹，否则为False
            pred = model(im, augment=augment, visualize=visualize)  # 推理，model()函数用于推理，im为输入图片，augment为是否使用数据增强，visualize为是否可视化,输出pred为一个列表，形状为（n,6）,n代表预测框的数量，6代表预测框的坐标和置信度，类别
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 准备属于这张图像的List容器
        the_pic_data = []
        # Process predictions  这个 for 从图像中反复识别目标，直到没有目标
        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    labelName = names[int(cls)]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    the_pic_data.append({
                        "label_name": labelName,
                        "xywh": xywh
                    })
        result_data.append(the_pic_data)
    print(*result_data)


if __name__ == "__main__":
    run(source=Path('./data/target'), conf_thres=0.75)
