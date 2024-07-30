## 该脚本用于在 PC 上使用摄像头进行实时目标检测
## 请确保已经安装了 torch 和 opencv-python 库
## 请确保已经下载了 YOLOv5 模型的权重文件
## the problem is that the hsv and grade switch is not working, example: when mouse at the orange color should be 40 degree but is 64. date:2024/7/30 02:18
import torch
import cv2
import numpy as np

# 使用绝对路径加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/geo/yolov5/runs/train/exp1record/weights/best.pt')

# 打开笔记本的摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 循环读取摄像头的每一帧
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # 转换图像为 RGB 格式
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 运行 YOLOv5 模型进行检测
    results = model(img_rgb)
    
    # 处理并显示检测结果
    for result in results.xyxy[0].cpu().numpy():  # 将张量移动到 CPU 并转换为 NumPy 数组
        # 解析检测框位置和置信度
        x1, y1, x2, y2, confidence, cls = result
        # 绘制检测框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 在检测框上方显示类别和置信度
        label = f"{model.names[int(cls)]}: {confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 显示检测后的图像
    cv2.imshow('YOLOv5 Detection', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()

