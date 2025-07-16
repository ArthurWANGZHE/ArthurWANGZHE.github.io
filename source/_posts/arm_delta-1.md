---
title: Delta机械臂（第一版）
date: 2025-07-16 15:16:48
series: 机械臂
tags:
---
# Delta机械臂（第一版，Python Qt）项目详解

## 项目背景

本项目为Delta机械臂的初步实现，主要目标是完成机械臂的基本运动控制、上下位机通信和仿真。采用Python和PyQt开发上位机，集成了视觉识别和多线程处理。

## 技术方案

- 语言/框架：Python、PyQt、多线程、OpenCV
- 功能模块：
  - 机械臂正逆运动学
  - 轨迹插补与运动规划
  - 上位机界面与多线程
  - 机械臂控制与数据通信
  - 视觉识别与目标检测

---

## 多线程机制与实现

### 理论说明

在机械臂上位机开发中，多线程机制的引入主要为了解决以下问题：

- **界面不卡顿**：机械臂控制、数据通信、视觉处理等任务耗时较长，若全部在主线程执行，会导致界面无响应。
- **任务并发**：需要同时进行运动控制、视觉识别、数据采集等多项任务。
- **信号与槽机制**：PyQt的多线程结合信号槽机制，可以安全地在不同线程间传递数据和事件。

### 关键代码实现

#### 1. 多线程任务封装

`worker_thread.py` 通过继承 `QThread`，将机械臂控制、通信等耗时操作放入子线程执行：

```python
from PyQt5.QtCore import QThread, pyqtSignal
from arm_controller import ArmController

class WorkerThread(QThread):
    armcontroller = ArmController()
    def __init__(self, method_name, *args, **kwargs):
        super().__init__()
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        # 信号连接
        WorkerThread.armcontroller.x_changed.connect(self.xxx)
        # ... 省略其他信号 ...

    def run(self):
        # 动态调用方法
        if hasattr(WorkerThread.armcontroller, self.method_name):
            getattr(WorkerThread.armcontroller, self.method_name)(*self.args, **self.kwargs)
        # ... 省略通信对象调用 ...
```

**说明：**

- 通过 `run()` 方法自动执行传入的方法名，实现任务的灵活分发。
- 通过信号槽机制，将子线程中的数据变化实时反馈到主界面，保证界面数据同步且不卡顿。

---

## 视觉识别机制与实现

### 理论说明

视觉模块主要用于机械臂末端或工件的识别与定位，常见流程包括：

1. **摄像头采集**：实时获取图像流。
2. **图像预处理**：如灰度化、HSV空间转换等。
3. **特征提取与匹配**：如SIFT、模板匹配等方法识别目标。
4. **目标标记与反馈**：在图像上绘制识别结果，并将位置信息用于后续控制。

### 关键代码实现

#### 1. 摄像头采集与多线程显示

`camera.py` 通过 `QThread` 和 `QTimer` 实现摄像头的实时采集与界面显示：

```python
class Camera(QObject):
    sendPicture = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super(Camera, self).__init__(parent)
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.timer = QTimer()
        self.init_timer()
        self.cap = cv2.VideoCapture()
        self.camera_num = 0

    def init_timer(self):
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.display)

    def open_camera(self):
        self.cap.set(4, 480)
        self.cap.set(3, 640)
        self.cap.open(self.camera_num)
        self.timer.start()
        self.thread.start()

    def display(self):
        flag, image = self.cap.read()
        image = self.match(image)
        showImage = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                 QtGui.QImage.Format_RGB888).rgbSwapped()
        self.sendPicture.emit(showImage)
```

**说明：**

- 采集与处理在子线程中进行，避免阻塞主界面。
- 通过 `sendPicture` 信号将处理后的图像实时发送到界面显示控件。

#### 2. 颜色与形状识别

```python
def match(self, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义红色和黄色的HSV范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    contours_red, _ = cv2.findContours(mask_red1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制识别结果
    for contour in contours_red:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 20, (255, 0, 0), 2)
    for contour in contours_yellow:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            triangle = np.array([[[cX - 20, cY], [cX + 20, cY], [cX, cY - 40]]], dtype=np.int32)
            cv2.polylines(image, triangle, True, (255, 0, 0), 2)
    return image
```

**说明：**

- 通过HSV空间分割红色、黄色区域，结合轮廓检测与几何中心计算，实现对目标的识别与标记。
- 可扩展为目标定位、抓取等更复杂的视觉任务。

---

## 关键代码片段（运动学与控制）

### 1. 运动学核心算法

```python
class DeltaRobotKinematics:
    def __init__(self, static_dia, moving_dia, link_length, length):
        self.static_dia = static_dia
        self.moving_dia = moving_dia
        self.link_length = link_length
        self.travel_range = (0, length)
        self.base_radius = static_dia / 2  # 静平台半径
        self.top_radius = moving_dia / 2   # 动平台半径
        # ... 省略部分初始化 ...

    def inverse_kinematics(self, t1, t2, t3):
        """
        逆运动学：根据滑块距离计算动平台中心坐标
        """
        # ... 复杂的三角几何与向量运算 ...
        return x, y, z

    def forward_kinematics(self, x, y, z):
        """
        正运动学：根据动平台中心坐标计算滑块距离
        """
        R = self.base_radius - self.top_radius
        l = self.link_length
        t3 = -z - np.sqrt(l ** 2 - x ** 2 - (y + R) ** 2)
        t2 = -z - np.sqrt(l ** 2 - (x - np.sqrt(3) / 2 * R) ** 2 - (R / 2 - y) ** 2)
        t1 = -z - np.sqrt(l ** 2 - (np.sqrt(3) / 2 * R + x) ** 2 - (R / 2 - y) ** 2)
        return [t1, t2, t3]
```

### 2. 轨迹插补与运动规划

```python
def point2point_ss(self, x1, y1, z1, x2, y2, z2):
    # 轨迹插补参数初始化
    # ... 计算加速度、速度、加加速度等参数 ...
    # 轨迹点生成
    # ... 生成平滑的插补轨迹点 ...
    return points, velocity, acceleration, jerk
```

### 3. 上位机界面与多线程

```python
from PyQt5 import QtCore, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(952, 512)
        self.com = QtWidgets.QComboBox(Widget)
        # ... 省略界面控件布局 ...
        self.pushButton = QtWidgets.QPushButton(Widget)
        self.cameraview = QtWidgets.QLabel(Widget)
        # ... 省略其他控件 ...
```

### 4. 机械臂控制与通信

```python
class ArmController(QObject):
    def move_arm_x(self, distance):
        self.communication.send_data("FF 40 FE")
        x1, y1, z1 = self.robot.xyz[0], self.robot.xyz[1], self.robot.xyz[2]
        self.robot.xyz[0] += distance
        x2, y2, z2 = self.robot.xyz
        points, velocity, acceleration, jerk = self.robot.point2point(x1, y1, z1, x2, y2, z2)
        packages = self.communication.packing(points, velocity, acceleration, jerk)
        self.communication.send_package(packages)
        t_new = self.robot.forward_kinematics(*self.robot.xyz)
        self.robot.t = t_new
        self.up_xyz()
        self.up_t()
```

## 项目反思

由于是初次尝试，代码结构较为混乱，功能实现以“能用”为主，后续有较大优化空间。
