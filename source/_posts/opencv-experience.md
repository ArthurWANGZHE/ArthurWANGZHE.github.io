---
title: opencv 面经
date: 2025-07-09 09:20:54
tags:
---


>  之前面试时针对opencv提了许多问题，也有手撕代码的环节做的不是很好。回忆了一下围绕opencv的问题，以及结合了一下自己过去的使用经验

# 模块分类

OpenCV 是一个模块化设计的计算机视觉库，其功能涵盖了从基础图像处理到三维重建、深度学习推理等多个层级。了解各模块的作用，有助于我们快速定位所需功能、查找文档，也更方便写代码时模块化组织。

下面是对 OpenCV 常见模块的简要分类：

## 基础功能模块

| 模块          | 说明                                                                         |
| ------------- | ---------------------------------------------------------------------------- |
| `core`      | 提供基础的数据结构（如 `cv::Mat`）、矩阵运算、随机数、排序、常用数学函数等 |
| `imgproc`   | 图像处理核心模块：滤波、边缘检测、几何变换、颜色空间转换等                   |
| `highgui`   | 图像/视频的图形界面显示、窗口创建、键盘交互                                  |
| `imgcodecs` | 图像文件读写支持（JPG、PNG、BMP、TIFF 等）                                   |
| `videoio`   | 视频流输入输出，包括摄像头/视频文件读写                                      |

## 视觉与算法模块

| 模块           | 说明                                         |
| -------------- | -------------------------------------------- |
| `features2d` | 特征检测与匹配（如 SIFT、ORB、FAST、BRIEF）  |
| `calib3d`    | 相机标定、三维重建、立体视觉、投影矩阵计算等 |
| `video`      | 视频分析，包括背景建模、光流估计、目标追踪   |
| `objdetect`  | 对象检测，如人脸检测（Haar、HOG等）          |
| `ml`         | 传统机器学习方法（如 SVM、KNN、决策树等）    |

## 深度学习与高级模块

| 模块          | 说明                                                          |
| ------------- | ------------------------------------------------------------- |
| `dnn`       | 加载并运行深度学习模型（支持 Caffe、TensorFlow、ONNX 等格式） |
| `photo`     | 图像增强、去噪、图像修复、无缝克隆等                          |
| `stitching` | 全景拼接，基于特征匹配 + 图像融合                             |
| `gapi`      | 现代图计算 API，适用于复杂管线的性能优化                      |

# 常用函数

## 图像读取与显示

| 功能         | 函数               | 说明                                    |
| ------------ | ------------------ | --------------------------------------- |
| 读取图像     | `cv2.imread()`   | 加载本地图像文件                        |
| 显示图像     | `cv2.imshow()`   | 弹出窗口显示图像                        |
| 等待键盘输入 | `cv2.waitKey()`  | 等待键盘事件，一般和 `imshow`搭配使用 |
| 保存图像     | `cv2.imwrite()`  | 将图像保存到文件                        |
| 改变图像尺寸 | `cv2.resize()`   | 可缩放图像，保持/不保持比例             |
| 色彩空间转换 | `cv2.cvtColor()` | 如 BGR ⇄ RGB ⇄ GRAY ⇄ HSV 等         |

**注：** 其中图像尺寸变换以及色彩空间转换的具体说明

```python
import cv2
# resized = cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR)
resized = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)  # 缩小图像
# 等比例缩放
scale = 0.5
h, w = img.shape[:2]
resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
```

* `src`：原始图像
* `dsize`：目标尺寸（宽, 高），如 `(640, 480)`
* `interpolation`：插值方法：
  * `cv2.INTER_NEAREST`（最近邻，最快）
  * `cv2.INTER_LINEAR`（双线性，默认）
  * `cv2.INTER_AREA`（图像缩小）
  * `cv2.INTER_CUBIC` / `cv2.INTER_LANCZOS4`（更高质量）

```python
import cv2
# dst = cv2.cvtColor(src, code)
img = cv2.imread("image.jpg")  # 读取彩色图像（默认 BGR）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 转为 RGB
```

* `src`：输入图像（如 BGR 图像）
* `code`：转换标志，如 `cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2HSV` 等
* 返回值：转换后的图像

## 摄像头模块

| 功能            | 函数                   | 说明               |
| --------------- | ---------------------- | ------------------ |
| 打开摄像头/视频 | `cv2.VideoCapture()` | 支持多种设备与格式 |
| 读取帧          | `cap.read()`         | 获取每一帧图像     |
| 写入视频        | `cv2.VideoWriter()`  | 保存帧序列为视频   |
| 设置参数        | `cap.set()`          | 设置帧率、分辨率等 |

**注：**`cv2.VideoCapture()` 的参数是index或者path，一般电脑自带的摄像头是0，使用usb摄像头时使用1或者其他

```python
cv2.VideoCapture(0)                 # 打开默认摄像头
cv2.VideoCapture("video.mp4")      # 打开本地视频文件
cv2.VideoCapture("rtsp://...")     # 打开网络摄像头或流媒体
```

## 图像处理

| 功能         | 函数                                               | 说明                   |
| ------------ | -------------------------------------------------- | ---------------------- |
| 高斯模糊     | `cv2.GaussianBlur()`                             | 去噪、平滑图像         |
| 中值滤波     | `cv2.medianBlur()`                               | 去除椒盐噪声           |
| 均值滤波     | `cv2.blur()`                                     | 基本模糊处理           |
| 边缘检测     | `cv2.Canny()`                                    | 边缘提取（需先模糊）   |
| 阈值分割     | `cv2.threshold()`                                | 固定/OTSU二值化        |
| 自适应阈值   | `cv2.adaptiveThreshold()`                        | 光照变化大的图像二值化 |
| 图像形态学   | `cv2.erode()`,`cv2.dilate()`                   | 腐蚀、膨胀、开闭运算等 |
| 图像旋转     | `cv2.getRotationMatrix2D()`+`cv2.warpAffine()` | 实现任意角度旋转       |
| 图像仿射变换 | `cv2.getAffineTransform()`                       | 三点变换               |
| 透视变换     | `cv2.getPerspectiveTransform()`                  | 四点变换               |

## 特征提取与匹配

| 功能         | 函数                                            | 说明                   |
| ------------ | ----------------------------------------------- | ---------------------- |
| 创建 ORB     | `cv2.ORB_create()`                            | 特征描述子             |
| 创建 SIFT    | `cv2.SIFT_create()`                           | 精度高但需额外安装     |
| 特征检测     | `detect()`或 `detectAndCompute()`           | 获取关键点和描述子     |
| 匹配特征     | `cv2.BFMatcher()`/`cv2.FlannBasedMatcher()` | 暴力匹配或快速匹配     |
| 单应矩阵估计 | `cv2.findHomography()`                        | 多点匹配后进行图像对齐 |
| 绘制匹配     | `cv2.drawMatches()`                           | 可视化匹配对结果       |

**注:** “算子”是 OpenCV 中用于提取图像关键点（keypoints）和生成描述子（descriptors）的方法的统称。

opencv中有有特征检测算子以及特征描述算子，检测器输入图像，输出关键点的合集，这些关键点本身只是坐标点，不携带“识别能力”。描述器输入图像+关键点，输出图像的特征向量

## 物体检测

| 功能             | 函数                                     | 说明                              |
| ---------------- | ---------------------------------------- | --------------------------------- |
| 加载 Haar 分类器 | `cv2.CascadeClassifier()`              | 如人脸检测                        |
| 使用深度网络     | `cv2.dnn.readNetFromONNX()`            | 加载 DNN 模型进行检测             |
| 跟踪目标         | `cv2.TrackerKCF_create()`              | 通过 `cv2.legacy`调用多种跟踪器 |
| 光流跟踪         | `cv2.calcOpticalFlowPyrLK()`           | 跟踪特征点的运动                  |
| 背景建模         | `cv2.createBackgroundSubtractorMOG2()` | 前景提取（运动检测）              |

# 手撕代码

## 涉及模块与函数

| 模块      | 功能               | 关键函数                                             |
| --------- | ------------------ | ---------------------------------------------------- |
| Video I/O | 摄像头输入         | `cv2.VideoCapture`                                 |
| 图像处理  | 灰度转换、边缘检测 | `cv2.cvtColor`, `cv2.Canny`                      |
| 特征提取  | ORB 特征           | `cv2.ORB_create`, `detectAndCompute`             |
| 特征匹配  | 描述子匹配         | `cv2.BFMatcher`, `match`                         |
| 可视化    | 绘图与展示         | `cv2.drawMatches`, `cv2.imshow`, `cv2.waitKey` |

## 项目流程

1. 打开摄像头；
2. 捕获第一帧作为“目标模板”；
3. 对后续每帧图像执行：
   - 灰度处理 + 边缘检测；
   - 使用 ORB 提取特征；
   - 与第一帧进行特征匹配；
   - 将匹配结果与边缘图进行可视化；
4. 用户按下 `q` 退出。

## 代码

```python
import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
orb = cv2.ORB_create()

# 捕获第一帧，作为目标模板
ret, first_frame = cap.read()
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(gray_first, None)

# 创建特征匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 灰度转换 + 边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # ORB 特征提取
    kp2, des2 = orb.detectAndCompute(gray, None)

    # 匹配特征（仅当描述子有效）
    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        match_img = cv2.drawMatches(first_frame, kp1, frame, kp2, matches[:20], None, flags=2)
    else:
        match_img = frame.copy()

    # 展示结果
    cv2.imshow("Edges", edges)
    cv2.imshow("Matches", match_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

<style>#mermaid-1752024977203{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1752024977203 .error-icon{fill:#552222;}#mermaid-1752024977203 .error-text{fill:#552222;stroke:#552222;}#mermaid-1752024977203 .edge-thickness-normal{stroke-width:2px;}#mermaid-1752024977203 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1752024977203 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1752024977203 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1752024977203 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1752024977203 .marker{fill:#333333;}#mermaid-1752024977203 .marker.cross{stroke:#333333;}#mermaid-1752024977203 svg{font-family:sans-serif;font-size:16px;}#mermaid-1752024977203 .label{font-family:sans-serif;color:#333;}#mermaid-1752024977203 .label text{fill:#333;}#mermaid-1752024977203 .node rect,#mermaid-1752024977203 .node circle,#mermaid-1752024977203 .node ellipse,#mermaid-1752024977203 .node polygon,#mermaid-1752024977203 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1752024977203 .node .label{text-align:center;}#mermaid-1752024977203 .node.clickable{cursor:pointer;}#mermaid-1752024977203 .arrowheadPath{fill:#333333;}#mermaid-1752024977203 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1752024977203 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1752024977203 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1752024977203 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1752024977203 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1752024977203 .cluster text{fill:#333;}#mermaid-1752024977203 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1752024977203:root{--mermaid-font-family:sans-serif;}#mermaid-1752024977203:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1752024977203 flowchart{fill:apa;}</style>
