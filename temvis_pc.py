import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

# 路径
visible_image_path = '/home/geo/Documents/detect/visualimage.jpeg'
thermal_image_path = '/home/geo/Documents/detect/hotimage.png'
yolo_model_path = '/home/geo/yolov5/runs/train/exp1record/weights/best.pt'

# 读取可见光图像和热图
visible_image = cv2.imread(visible_image_path)
thermal_image = cv2.imread(thermal_image_path)

# 调整热图分辨率以匹配可见光图像
thermal_image_resized = cv2.resize(thermal_image, (visible_image.shape[1], visible_image.shape[0]))

# 图像配准（使用特征点匹配）
def align_images(thermal_image, visible_image):
    # 将图像转换为灰度图
    gray_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    gray_visible = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)
    
    # 使用ORB检测关键点和描述符
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_thermal, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_visible, None)
    
    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 提取匹配的关键点
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 计算透视变换矩阵
    matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    
    # 对热图进行透视变换
    aligned_thermal = cv2.warpPerspective(thermal_image, matrix, (visible_image.shape[1], visible_image.shape[0]))
    return aligned_thermal

aligned_thermal_image = align_images(thermal_image_resized, visible_image)

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)

# 运行 YOLOv5 模型进行检测
results = model(cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB))

# 定义一个函数将灰度值转换为温度
min_temp = 28.0
max_temp = 63.6

def gray_to_temp(gray_value):
    return min_temp + (max_temp - min_temp) * (gray_value / 255.0)

# 提取检测框中心点的温度信息
def get_temperature_at_center(thermal_image, box, size=5):
    x1, y1, x2, y2 = map(int, box[:4])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    roi = thermal_image[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    gray_values = np.mean(roi, axis=(0, 1))
    gray_value = np.mean(gray_values)
    temperature = gray_to_temp(gray_value)
    return temperature

# 显示检测结果和温度信息
output_info = []
for result in results.xyxy[0]:
    x1, y1, x2, y2, confidence, cls = result.cpu().numpy()
    temperature = get_temperature_at_center(aligned_thermal_image, (x1, y1, x2, y2))
    
    # 绘制检测框和温度信息
    cv2.rectangle(visible_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{model.names[int(cls)]}: {confidence:.2f}, Temp: {temperature:.2f}C"
    cv2.putText(visible_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 添加到输出信息列表
    output_info.append(f"{model.names[int(cls)]}: {temperature:.2f}C")

# 显示融合后的图像
plt.imshow(cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Components with Temperature')
plt.show()

# 输出结果
for info in output_info:
    print(info)
