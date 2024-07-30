from picamera2 import Picamera2, controls
import torch
import cv2
import numpy as np
import time

# 使用绝对路径加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/team2/cam_test_20240729/exp1record/weights/best.pt')

# 打开摄像头
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)
picam2.start()

# 等待摄像头启动
time.sleep(2)

frame_count = 0  # 初始化帧计数器
last_results = None  # 保存上一次的检测结果

################################
while True:
    # 捕获帧作为 numpy 数组, RGB 格式
    img_rgb = picam2.capture_array()

    # 转换图像为 BGR 格式，以便在 OpenCV 中显示
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 每5帧进行一次检测
    if frame_count % 5 == 0:
        # 运行 YOLOv5 模型进行检测
        results = model(img_rgb)
        last_results = results.xyxy[0].cpu().numpy()  # 更新检测结果

    # 使用最后一次的检测结果
    if last_results is not None:
        # 处理并显示检测结果
        for result in last_results:  # 使用保存的检测结果
            # 解析检测框位置和置信度
            x1, y1, x2, y2, confidence, cls = result
            # 绘制检测框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 在检测框上方显示类别和置信度
            label = f"{model.names[int(cls)]}: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示检测后的图像
    cv2.imshow('YOLOv5 Detection', frame)
    frame_count += 1  # 增加帧计数器

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
################################


# 停止摄像头并关闭所有窗口
picam2.stop()
cv2.destroyAllWindows()
