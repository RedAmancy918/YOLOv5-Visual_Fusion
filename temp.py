import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = '/home/geo/Documents/detect/hotimage.png'
image = cv2.imread(image_path)

# 图例部分的坐标和温度范围，需要根据实际情况调整
legend_top_left = (820, 50)  # 图例左上角坐标
legend_bottom_right = (840, 420)  # 图例右下角坐标
min_temp = 28.0  # 图例显示的最小温度
max_temp = 63.6  # 图例显示的最大温度

# 提取图例区域的图像
legend_image = image[legend_top_left[1]:legend_bottom_right[1], legend_top_left[0]:legend_bottom_right[0]]

# 假设从蓝色（低温）到红色（高温）的颜色范围
color_to_temp = {
    (255, 0, 0): min_temp,  # 蓝色
    (0, 0, 255): max_temp   # 红色
}

# 插值函数，根据颜色值获取温度
def interpolate_temp(color):
    min_color = np.array([255, 0, 0])
    max_color = np.array([0, 0, 255])
    min_temp = 28.0
    max_temp = 63.6
    ratio = np.linalg.norm(color - min_color) / np.linalg.norm(max_color - min_color)
    return min_temp + ratio * (max_temp - min_temp)

# 提取任意像素点的温度
def get_temperature_at_pixel(image, x, y):
    pixel_value = image[y, x]
    temperature = interpolate_temp(pixel_value)
    return temperature

# 显示图像并添加交互功能
def show_image_with_temperature(image):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Click to get temperature')

    def onclick(event):
        x, y = int(event.xdata), int(event.ydata)
        if x is not None and y is not None:
            temperature = get_temperature_at_pixel(image, x, y)
            print(f'Temperature at ({x}, {y}): {temperature:.2f} °C')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# 运行函数显示图像并获取温度
show_image_with_temperature(image)

