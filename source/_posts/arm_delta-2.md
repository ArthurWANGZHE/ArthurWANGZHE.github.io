---
title: Delta机械臂(第二版)
date: 2025-07-16 15:18:05
series: 机械臂
tags:
---
# Delta机械臂（C++ Qt + 运动控制卡）项目详解

## 项目背景

在第一版基础上，采用C++ Qt重构上位机，配合运动控制卡实现高精度运动控制，系统功能更加完善。实现了白板轨迹示教等特色功能。

## 技术方案

- 语言/框架：C++、Qt、运动控制卡SDK
- 功能模块：
  - 机械臂正逆运动学
  - 轨迹插补与运动规划
  - 上位机界面与白板轨迹示教
  - 运动控制卡通信与实时控制

---

## 白板示教功能逻辑与实现

### 理论说明

白板示教是本系统的特色功能，允许用户在上位机界面“白板”上手绘轨迹，系统自动采集轨迹点并转化为机械臂运动指令，实现“所画即所得”的轨迹复现。其核心流程包括：

1. **轨迹采集**：通过鼠标事件实时记录用户在白板上的手绘路径。
2. **轨迹处理**：对采集到的点进行坐标变换、降噪、插值等处理，转化为机械臂工作空间的轨迹点。
3. **轨迹下发**：将轨迹点依次发送给运动控制卡，驱动机械臂复现手绘轨迹。

### 关键代码实现

#### 1. 轨迹采集与处理（WhiteboardControl.cs）

```csharp
internal class WhiteboardControl : Panel
{
    private bool isDrawing = false;
    private List<Point> points = new List<Point>();  // 当前路径
    private List<List<Point>> allPaths = new List<List<Point>>();  // 所有路径

    public WhiteboardControl()
    {
        this.DoubleBuffered = true;
        this.BackColor = Color.White;
        this.MouseDown += WhiteboardControl_MouseDown;
        this.MouseMove += WhiteboardControl_MouseMove;
        this.MouseUp += WhiteboardControl_MouseUp;
    }

    // 鼠标按下事件，开始绘制
    private void WhiteboardControl_MouseDown(object sender, MouseEventArgs e)
    {
        isDrawing = true;
        points.Clear();
        points.Add(e.Location);
    }

    // 鼠标移动事件，记录坐标并绘制
    private void WhiteboardControl_MouseMove(object sender, MouseEventArgs e)
    {
        if (isDrawing)
        {
            allPaths.Add(new List<Point>(points));
            points.Add(e.Location);
            this.Invalidate();
        }
    }

    // 鼠标抬起事件，结束绘制并保存轨迹
    private void WhiteboardControl_MouseUp(object sender, MouseEventArgs e)
    {
        isDrawing = false;
        if (points.Count > 0)
        {
            allPaths.Add(new List<Point>(points));
        }
    }

    // 获取轨迹点并转为机械臂坐标
    public List<(double x, double y, double z)> get_paths()
    {
        int rowCount = allPaths.Count;
        List<(double x, double y, double z)> coordinates = new List<(double, double, double)>();
        double x = allPaths[rowCount - 1][0].X / 2;
        double y = allPaths[rowCount - 1][0].Y / 2;
        x = x - 85;
        y = 85 - y;
        coordinates.Add((x, y, 60));
        for (int i = 1; i < rowCount; i++)
        {
            var (x0, y0, z0) = coordinates[coordinates.Count - 1];
            double x1 = allPaths[rowCount - 1][i - 1].X / 2;
            double y1 = allPaths[rowCount - 1][i - 1].Y / 2;
            x1 = x1 - 85;
            y1 = 85 - y1;
            double distance = Math.Sqrt(Math.Pow(x1 - x0, 2) + Math.Pow(y1 - y0, 2));
            if (distance > 1)
            {
                coordinates.Add((x1, y1, 60));
            }
        }
        return coordinates;
    }
}
```

**说明：**

- 通过鼠标事件采集用户手绘轨迹，支持多段轨迹。
- `get_paths()` 方法将屏幕坐标转换为机械臂工作空间坐标，并做距离过滤，保证轨迹平滑。

---

## 运动控制卡调用与实现

### 理论说明

运动控制卡负责将上位机下发的轨迹点转化为具体的电机控制信号，实现机械臂的精准运动。调用流程包括：

1. **参数设置**：设置加速度、速度等运动参数。
2. **运动指令下发**：调用控制卡SDK接口，发送绝对/相对运动、插补等指令。
3. **状态反馈**：通过SDK接口获取运动状态，实现闭环控制。

### 关键代码实现（Gantry.cs）

```csharp
// 绝对位置运动
private void Button5_Click(object sender, EventArgs e)
{
    int axis = ComboBox1.SelectedIndex;
    double acc = Convert.ToDouble(TextBox8.Text);
    double startvel = Convert.ToDouble(TextBox7.Text);
    double endvel = Convert.ToDouble(TextBox6.Text);
    double tgvel = Convert.ToDouble(textBox3.Text);
    int pos = Convert.ToInt32(textBox4.Text);

    // 设置加速度
    int st = IMC_Pkg.PKG_IMC_SetAccel(Global.g_handle, acc, acc, axis);
    if (st != 0) {
        // 绝对位置运动
        st = IMC_Pkg.PKG_IMC_MoveAbs(Global.g_handle, pos, startvel, endvel, tgvel, 0, axis);
    }
    if (st == 0) {
        string err = Global.GetFunErrStr();
        MessageBox.Show(err);
    }
}
```

**说明：**

- 通过 `IMC_Pkg.PKG_IMC_SetAccel` 设置加速度参数。
- 通过 `IMC_Pkg.PKG_IMC_MoveAbs` 实现绝对位置运动，参数包括目标位置、速度、加速度等。
- 运动控制卡SDK接口丰富，支持多种运动模式（点到点、插补、联动等）。

---

## 项目亮点

- 白板示教：用户可在白板上手绘轨迹，机械臂自动复现。
- 运动控制卡：实现高精度、实时的运动控制。
- 系统功能完整，用户体验大幅提升。
