"""
超大行李智能识别v2.0 —— 摄像头模式
"""
import torch
import time
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (check_img_size,increment_path, cv2,non_max_suppression, scale_boxes, xyxy2xywh,)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
@smart_inference_mode()  # 用于自动切换模型的推理模式，如果是FP16模型，则自动切换为FP16推理模式，否则切换为FP32推理模式，这样可以避免模型推理时出现类型不匹配的错误
async def run(
        weights="weight/date121size8epochs285.pt",
        imgsz=(640, 640),  # 初始尺寸(height, width)
        conf_thres=0.45,  # 可信阈值
        iou_thres=0.45,  # 合并阈值
        project= 'runs/detect',  # 结果存放路径
        name='exp',  # 结果文件夹名称
        line_thickness=3,  # 绘制框宽度
        person_height=170,  # 图像中参考的人物高度 cm
        real_height_dec=10,  # 误差高度
        websocket=None
):
    """
    - 接收人物高度参数
    """
    # Directories，创建保存结果的文件夹
    save_dir = increment_path(Path(project) / name,exist_ok=False)
    save_dir.mkdir(parents=True,exist_ok=True)  # make dir，创建save_dir文件夹

    # 初始化
    device = select_device()  # 选择设备
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型的步幅、名称以及类型。步幅更小提高精度，更大减小性能开销
    imgsz = check_img_size(imgsz, s=stride)  # 验证图像大小是每个维度的stride=32的倍数
    dataset = LoadStreams("0", img_size=imgsz, stride=stride)
    model.warmup(imgsz=(1, 3, *imgsz))  # 预热，用于提前加载模型，加快推理速度
    vid_path, vid_writer = [None] * 1, [None] * 1

    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集，path为图片路径，im为图片，im0s为原始图片，vid_cap为视频读取对象，s为视频帧率

        im = torch.from_numpy(im).to(model.device)  # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32    #如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
        im /= 255  # 0 - 255 to 0.0 - 1.0    #将图片归一化，将图片像素值从0-255转换为0-1

        if len(im.shape) == 3:  # 图像通常有三个维度 (height, width, channels（通道数）)
            im = im[None]  # #在前面添加batch维度（批处理维度），pytorch模型的输入必须是4维的


        pred = model(im, augment=True) # 获得推理结果
        pred = non_max_suppression(pred, conf_thres, iou_thres)  # 去除重复的预测框

        the_pic_data = []  # 准备属于这帧图像的List容器

        for i, det in enumerate(pred):  # i为索引，det为每个物体的预测框

            im0 =  im0s[i].copy() #ims[i].copy()为将输入图像的副本存储在im0变量中
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) #标注工具对象

            if len(det):  # 如果存在预测框

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # 将预测框的坐标从归一化坐标转换为原始的真实坐标信息 （必要的）

                # 反转列表并循环
                for *xyxy, conf, cls in reversed(det):
                    the_pic_data.append({
                        "label_name":  names[int(cls)],
                        "xywh": (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / (torch.tensor(im0.shape)[[1, 0, 1, 0]])).view(-1).tolist(), # 坐标信息
                        "xyxy": xyxy  # 原始绘制信息
                    })

                max_h = 0 # 找到最person标签的最大值，用于计算图像现实中实际高度
                for i, item in enumerate(the_pic_data):
                    if item['label_name'] == 'person':
                        h = item['xywh'][3]
                        if h > max_h:
                            max_h = h

                # 如果帧中不包含人物，将无法计算行李的大小
                if max_h == 0 :
                    for i, item in enumerate(the_pic_data):
                        if item['label_name'] == 'luggage':
                            label = f"{item['label_name']}" + f", No one was found"
                            annotator.box_label(item['xyxy'], label, color=[255, 0, 0])

                # 包含人物，正确计算
                else:
                    pic_real_h = (person_height+10)/max_h  # 图像现实中实际高度
                    real_w, real_h = im0.shape[1], im0.shape[0]
                    size_scale = real_w / real_h  # 宽高比
                    pic_real_w = pic_real_h * size_scale
                    # xw 和宽度有关系， yh和高度有关系
                    for i, item in enumerate(the_pic_data):
                        p_xywh = item['xywh']
                        r_xywh = [p_xywh[0]*pic_real_w,p_xywh[1]*pic_real_h, p_xywh[2]*pic_real_w, (p_xywh[3]*pic_real_h)-real_height_dec]
                        item['r_xywh'] = r_xywh
                        if item['label_name'] == 'luggage':
                            isBig = r_xywh[2] >= 55 or r_xywh[3] >= 40       # 55厘米、40厘米 ,只要长高不超过，将忽略最短的宽边
                            label = f"{item['label_name']}" + f" w{r_xywh[2]:.2f}cm*h{r_xywh[3]:.2f}cm " + f"{'BigLuggage' if isBig else ''} "
                            annotator.box_label(item['xyxy'], label, color=[0,0,255] if isBig else [0,255,0])

            im0 = annotator.result()  # 获取图像
            # cv2.waitKey(1)  # 毫秒刷新
            # cv2.imshow("Recognizing, max size : 20x40x55cm", im0)  # 显示图像
            # 将图像编码为 JPEG 格式
            _, buffer = cv2.imencode('.jpg', im0)

            # 将结果转换为字节流
            jpg_as_bytes = buffer.tobytes()
            time.sleep(2)
            await websocket.send(jpg_as_bytes)

