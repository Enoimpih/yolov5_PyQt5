import time
import cv2
import os
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import colors, Annotator
from utils.torch_utils import select_device


def compute(img, min_percentile, max_percentile):
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def detect(
        weights='C:/Users/lenovo/Desktop/exp34/weights/best.pt',  # 训练好的模型路径   （必改）
        imgsz=512,  # 训练模型设置的尺寸
        conf_thres=0.25,  # 置信度
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 最大侦测的目标数
        device='',  # 设备
        crop=True,  # 显示预测框
        classes=None,  # 种类
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # 是否扩充推理
        half=False,  # 使用FP16半精度推理
        hide_labels=False,  # 是否隐藏标签
        hide_conf=True,  # 是否隐藏置信度
        line_thickness=3  # 预测框的线宽
):
    # -----初始化-----无需修改
    set_logging()
    device = select_device(device)  # 设置设备
    half &= device.type != 'cpu'  # CUDA仅支持半精度！
    # -----加载模型-----
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步幅
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
    names = model.module.names if hasattr(model, 'module') else model.names  # 获取类名
    # toFP16
    if half:
        model.half()  # ------运行推理------
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 跑一次

    t0 = time.time()
    count = 0
    tplay = 0

    path = 'C:/Users/lenovo/Desktop/01.mp4'    #视频路径
    cap = cv2.VideoCapture(path)
    while True:
        # 截取摄像头最新一帧为img0
        ret, img0 = cap.read()
        img0 = aug(img0)

        if True:
            # 目标检测过程-----无需修改-----
            labels = []  # 设置labels--记录标签/概率/位置
            img = letterbox(img0, imgsz, stride=stride)[0]  # 填充调整大小
            img = img[:, :, ::-1].transpose(2, 0, 1)  # 转换BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=augment)[0]  # 推断
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 添加 NMS

            # 目标进程
            for i, det in enumerate(pred):  # 每幅图像的检测率
                s, im1 = '', img0.copy()
                s += '%gx%g ' % img.shape[2:]  # 输出字符串
                gn = torch.tensor(im1.shape)[[1, 0, 1, 0]]  # 归一化增益
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im1.shape).round()    # 将框从img_大小重新缩放为im0大小
                    for c in det[:, -1].unique():  # 输出结果
                        n = (det[:, -1] == c).sum()  # 每类检测数
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串
                    if names[int(c)] == 'RB':   #判断标签类型为骑车
                        value = det[:, 4].max().item()  # 解决的是看到视频就报警 每取四个值 找到最大
                        print(value)
                        if value > 0.8:  # 大于0.8播放警示语音
                            count += 1  # count等于一个滤波 小延时 防止误报 累计5个数之后再报
                            if count > 5:
                                count = 0
                                if time.time() - tplay > 3:  # 防止声音叠加 1.8s播放完再继续
                                    os.system(
                                        'start /b D:/ffmpeg-master-latest-win64-gpl/bin/ffplay.exe -autoexit -nodisp C:/Users/lenovo/Desktop/02.mp3')  # 音乐播放
                                    # 参数含义： start /b 后台启动    ffplay音乐播放软件的位置
                                    # -autoexit 播放完毕自动退出    -nodisp不显示窗口     mp3语音的位置路径
                                    tplay = time.time()  # 当前系统时钟
                    for *xyxy, conf, cls in reversed(det):  # 结果输出
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
                        line = (cls, *xywh, conf)  # 标签格式
                        c = int(cls)  # 整数类
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 建立标签
                        annotator = Annotator(im1, line_width=line_thickness, example=str(names))  # 绘画预测框
                        if crop:
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        labels.append([names[c], conf, xyxy])  # 记录标签/概率/左上角坐标以及右下角坐标位置（使用）
            # 后续可通过对labels的读取获取目标物的名称，概率以及对应坐标，xyxy为目标物左上角坐标以及右下角坐标，转成int即可！
            print(labels)
            # 在此处进行交互
            L = len(labels)
            # 显示图片，im1为处理后结果
            cv2.imshow("result", im1)
            # 通过图片测试，让窗口长时间停留，跑视频时记得改一下！
            cv2.waitKey(10)


if __name__ == "__main__":
    detect()
