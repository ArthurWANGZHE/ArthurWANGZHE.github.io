---
title: 并联scara机械臂
date: 2025-07-16 15:20:27
series: 机械臂
tags:
---
# Scara机械臂系统（C++ Qt）项目详解

## 项目背景

针对Scara机械臂构型开发的完整系统，仿真与控制高度集成于上位机。实现了多种轨迹插补算法和基于视觉的拖动示教。

## 技术方案

- 语言/框架：C++、Qt、OpenCV
- 功能模块：
  - 机械臂正逆运动学
  - 多种轨迹插补算法（直线、圆弧、样条）
  - 上位机仿真与可视化
  - 视觉拖动示教

---

## 内置仿真与运动学实现

### 实现逻辑

- 系统内置了完整的Scara机械臂运动学模型（`ScaraKinematics`），支持正逆解计算。
- 通过Qt的绘图机制（如 `QPainter`、`QChart`等）实时渲染机械臂各连杆、关节、末端轨迹，实现仿真可视化。
- 用户可在界面上交互式设置目标点，系统自动计算关节角度并在仿真窗口动态显示机械臂运动过程。

### 关键代码片段

#### 1. 运动学正逆解（`ScaraKinematics.cpp/.h`）

```cpp
// 正向运动学：已知关节角，求末端位置
QPointF ScaraKinematics::forwardKinematics(float phi1_deg, float phi4_deg) {
    // ... 省略参数设置 ...
    // 计算B、D点
    block.scaraPoint[1].x = block.scaraPoint[0].x + block.link[0] * qCos(block.phi[0]);
    block.scaraPoint[1].y = block.scaraPoint[0].y + block.link[0] * qSin(block.phi[0]);
    // ... 计算中间点 ...
    // 计算末端点C
    block.scaraPoint[2].x = block.scaraPoint[1].x + block.link[1] * qCos(block.phi[1]);
    block.scaraPoint[2].y = block.scaraPoint[1].y + block.link[1] * qSin(block.phi[1]);
    return QPointF(block.scaraPoint[2].x, block.scaraPoint[2].y);
}

// 逆向运动学：已知末端位置，求关节角
QPointF ScaraKinematics::inverseKinematics(float x, float y) {
    // ... 省略参数设置 ...
    // 计算phi1、phi4
    // ... 逆解算法 ...
    block.scaraJoint.x = radToDeg(block.phi[0]);
    block.scaraJoint.y = radToDeg(block.phi[3]);
    return QPointF(block.scaraJoint.x, block.scaraJoint.y);
}
```

#### 2. 仿真可视化（`mainwindow.cpp`）

- 通过 `QChart`、`QLineSeries`等对象实时绘制机械臂轨迹和末端运动路径。
- 机械臂每次运动后，调用 `addPointOptimized(x, y)`将末端点加入轨迹，实现轨迹可视化。

---

## 视觉拖动示教功能

### 实现逻辑

- 系统集成了OpenCV摄像头采集，实时获取机械臂末端或示教笔的图像。
- 用户可通过拖动末端（如用手或标记物），系统通过图像处理（如颜色识别、形状检测等）获取末端像素坐标。
- 像素坐标通过 `pixelToWorld`函数转换为机械臂工作空间坐标，自动调用逆解算法，驱动机械臂跟随拖动轨迹，实现“视觉拖动示教”。

### 关键代码片段

#### 1. 摄像头采集与图像处理（`mainwindow.cpp`）

```cpp
// 摄像头初始化
cap.open(1);
if (!cap.isOpened()) {
    QMessageBox::critical(this, "错误", "无法打开摄像头");
    return;
}
cameraTimer = new QTimer(this);
connect(cameraTimer, &QTimer::timeout, this, &MainWindow::readFrame);
cameraTimer->start(30);

// 图像帧处理
void MainWindow::readFrame() {
    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) {
        // 图像处理：如颜色识别、形状检测，获取末端像素坐标
        // cv::findContours、cv::moments等
        // ...
        // 坐标转换
        cv::Point2f worldPos = pixelToWorld(pixelPos);
        // 逆解驱动机械臂
        QPointF jointAngles = m_ScaraKinematics->inverseKinematics(worldPos.x, worldPos.y);
        // 控制机械臂运动
        // ...
    }
}
```

#### 2. 像素到世界坐标转换

```cpp
cv::Point2f MainWindow::pixelToWorld(const cv::Point2f& pixel) {
    // 根据标定参数，将像素坐标转换为机械臂工作空间坐标
    // 例如：比例尺、偏移量等
    float x = (pixel.x - offsetX) * scaleX;
    float y = (pixel.y - offsetY) * scaleY;
    return cv::Point2f(x, y);
}
```

---

## 总结亮点

- **内置仿真**：运动学正逆解+Qt实时绘制，支持轨迹插补与多种运动模式，便于调试和演示。
- **视觉拖动示教**：OpenCV+Qt集成，支持通过摄像头拖动末端实现轨迹示教，极大提升人机交互体验。
- **工程集成度高**：仿真、控制、视觉一体化，适合展示综合工程能力。
