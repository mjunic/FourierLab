# FourierLab

FourierLab 是一个面向《信号分析与处理》课程展示的本地图像频域实验软件。软件标题为“FourierLab —— 二维傅里叶图像实验平台”，包含频域滤波、Hybrid Image 融合和傅里叶域图像加密三个实验 Tab。

详细操作教程见 [USER_GUIDE.md](USER_GUIDE.md)。

## 主要功能

- 打开 jpg、png、bmp 图片，默认转为灰度图处理。
- 可勾选彩色 RGB 处理，对 R、G、B 三个通道分别做 FFT、滤波和 IFFT。
- 自动缩放过大的图片，最大边不超过 1024，避免课堂演示时卡顿。
- 使用 `numpy.fft.fft2` 计算二维傅里叶变换。
- 使用 `numpy.fft.fftshift` 将低频移动到频谱中心。
- 显示幅度谱 `log(abs(F)+1)`、相位谱和滤波器模板。
- 支持基础频域滤波：
  - 不滤波
  - 理想低通
  - 理想高通
  - 理想带通
  - 高斯低通
  - 高斯高通
  - 巴特沃斯低通
  - 巴特沃斯高通
- 支持手动调节截止半径、带通内外半径、高斯 sigma、巴特沃斯阶数。
- 支持自动低通调参：按中心累计能量 80%、90%、95%、99% 自动选择半径。
- 支持 IFFT 重建和结果保存。
- 支持 3D 频谱曲面图。
- 支持学习引导文本，随操作步骤解释图像输入、FFT、滤波和 IFFT 重建原理。
- 支持导出四宫格实验图：原图、2D幅度谱、滤波器模板、重建图像。
- 支持 Hybrid Image 融合实验：A 图低频轮廓 + B 图高频细节。
- 支持教学型频域加密实验：扰乱傅里叶相位谱并使用相同 seed 尝试解密。
- 显示图片尺寸、频谱最大值、频谱平均值、低频/高频能量占比、保留能量比例、MSE 和 PSNR。

## 项目结构

```text
FourierLab/
├── main.py
├── requirements.txt
├── README.md
├── build_exe.bat
├── fourierlab/
│   ├── __init__.py
│   ├── app.py
│   ├── ui_main.py
│   ├── image_io.py
│   ├── fourier_core.py
│   ├── filters.py
│   ├── metrics.py
│   ├── hybrid.py
│   ├── encryption.py
│   └── visualization.py
└── examples/
    └── README.md
```

`hybrid.py` 和 `encryption.py` 保留为扩展模块，但当前主界面不展示这些功能，避免软件显得过杂。

## 安装依赖

建议 Python 3.10 或更高版本。

```powershell
cd C:\Users\mjuni\FourierLab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果不想创建虚拟环境，也可以直接：

```powershell
cd C:\Users\mjuni\FourierLab
pip install -r requirements.txt
```

## 运行方法

```powershell
cd C:\Users\mjuni\FourierLab
python main.py
```

Tab 1 频域滤波实验：

1. 点击“打开图片”。
2. 软件自动显示原图并计算 FFT。
3. 选择滤波器，例如“理想低通”或“理想高通”。
4. 调整参数滑块。
5. 点击“应用滤波并重建”查看处理后图像。
6. 点击“导出四宫格实验图”生成课程报告图片。
7. 点击“查看 3D 频谱”观察频谱曲面。

Tab 2 Hybrid Image 融合实验：

1. 导入图像 A，作为低频轮廓来源。
2. 导入图像 B 或文字图片，作为高频细节来源。
3. 调整 r_low、r_high、alpha、beta。
4. 点击“生成 Hybrid Image”。
5. 点击“保存融合结果”。

Tab 3 频域加密实验：

1. 导入图片。
2. 设置 seed 和相位扰动强度。
3. 点击“加密”生成加密图。
4. 点击“解密”使用相同 seed 尝试恢复。
5. 分别保存加密图和解密图。

## 打包 exe

```powershell
cd C:\Users\mjuni\FourierLab
.\build_exe.bat
```

脚本内命令为：

```bat
pyinstaller -F -w -n FourierLab main.py
```

如果打包后提示 PySide6 或 Matplotlib 资源缺失，可尝试：

```powershell
pyinstaller -F -w -n FourierLab --collect-all PySide6 --collect-all matplotlib main.py
```

## 傅里叶变换原理简述

二维图像可以看成空间域信号。傅里叶变换把图像分解为不同频率的正弦/余弦成分：

```python
F = np.fft.fft2(image)
F_shift = np.fft.fftshift(F)
spectrum = np.log(np.abs(F_shift) + 1)
```

频谱中心主要是低频，表示图像整体亮度和大轮廓；远离中心的区域主要是高频，表示边缘、纹理和噪声。

逆变换流程：

```python
filtered = F_shift * mask
image_back = np.fft.ifft2(np.fft.ifftshift(filtered))
```

## 滤波器说明

- 理想低通：保留中心圆形区域，图像会变平滑。
- 理想高通：去掉中心低频，突出边缘和细节。
- 理想带通：只保留一段频率范围。
- 高斯低通/高通：过渡更平滑，视觉效果通常比理想滤波更自然。
- 巴特沃斯低通/高通：通过阶数控制过渡陡峭程度。

## 展示建议

- 用一张人像或风景图展示低通滤波，说明低频代表整体轮廓。
- 用文字、线稿或边缘明显的图展示高通滤波，说明高频代表边缘细节。
- 演示不同半径下重建图像的变化。
- 打开 3D 频谱图，说明频谱中心能量通常更高。
- 展示 MSE 和 PSNR 如何随滤波强度变化。
