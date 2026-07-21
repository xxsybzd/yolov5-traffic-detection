# 🚦 基于 YOLOv5 的城市道路车辆类别与行人检测系统

本项目基于 **YOLOv5** 深度学习目标检测框架构建，针对城市交通场景下的 **小型汽车 (car)**、**大型车辆 (large_vehicle)**、**两轮车 (two_wheeler)** 及 **行人 (pedestrian)** 实现高效多目标精准识别与追踪。系统搭配 Streamlit 可视化 Web 界面，支持图片、视频及屏幕实时区域追踪。

---
## 📦 完整数据集下载

由于 GitHub 单文件与仓库体积限制，本仓库仅包含用于测试与展示的样例数据（位于 `1、traffic_dataset (样图)/`）。包含交通场景全量图片及 YOLO 格式标注信息的完整数据集已上传至百度网盘：

* **📥 下载链接**：[百度网盘 - 完整数据集 dataset_full.zip](https://pan.baidu.com/s/1h6t1uT0cfV2do7M4U-kLig?pwd=89vd)
* **🔑 提取码**：`89vd`

> **解压提示**：下载 `dataset_full.zip` 后解压，替换或解压至项目根目录下的 `1、traffic_dataset` 文件夹即可用于重新训练或大批量测试。
## ⚠️ 路径配置说明

## 📷 系统实际运行效果图

以下截图为系统在 Web 端实时推理与追踪的**实际运行效果图**（相关图片存于 `test_images/` 目录）：

| 效果图一 | 效果图二 | 效果图三|
| :---: | :---: | :---: |
| ![车辆检测效果图](test_images/test1.jpg) | ![行人检测效果图](test_images/test2.jpg) | ![屏幕实时追踪效果图](test_images/test3.jpg) |

---

## 🌟 核心功能亮点

1. **📸 静态图像 / 文件夹检测**：支持上传单张交通照片或批量导入图片文件夹，实现多目标特征提取与可视化框选。
2. **🎬 离线视频逐帧推理**：支持上传路口监控视频流，进行流畅的逐帧检测与渲染展示。
3. **🖥️ 屏幕任意区域实时追踪**：支持自定义框选电脑屏幕的物理区域（如播放网页视频、监控直播流），实现无缝高帧率同步推理。

---

## 📁 项目目录结构

```text
yolov5-traffic-detection/
├── 1、traffic_dataset (样图)/ # 交通场景训练/测试数据集样例
├── 2、配置文件/                # 数据集配置文件 (traffic_data.yaml) 与模型结构配置
├── 3、训练脚本/                # 模型训练脚本 (train.py) 与 Web 交互主程序 (web_test.py)
├── 4、权重文件/                # 训练好的模型权重 (.pt) 及性能评价曲线图
├── test_images/              # 存放系统实际运行效果截图 (test1~3.jpg)
├── .gitignore                # Git 忽略文件配置
├── LICENSE                   # 开源许可协议 (MIT)
├── README.md                 # 项目说明文档
└── requirements.txt          # 项目依赖库清单


由于本项目部分脚本与配置文件（如 `2、配置文件/traffic_data.yaml` 和 `3、训练脚本/web_test.py`）包含开发环境的本地绝对路径，**在他人机器上运行前请注意进行以下微调**：

1. **修改数据集路径**：
   打开 `2、配置文件/traffic_data.yaml`，将 `path` 修改为你本地项目解压后的绝对路径或相对路径：
   ```yaml
   path: G:/桌面/深度学习项目/.../1_traffic_dataset  # 改为你本地的实际路径
   train: images/train
   val: images/val
   ```

2. **修改权重文件/脚本加载路径**：
   若在 `web_test.py` 中提示找不到模型权重或路径报错，请打开文件将开头的 `weights_path` 或 `project_root` 修改为你本地的实际存储路径。
## 🛠️ 快速上手

### 1. 克隆项目与安装依赖
```bash
git clone [https://github.com/xxsybzd/yolov5-traffic-detection.git](https://github.com/xxsybzd/yolov5-traffic-detection.git)
cd yolov5-traffic-detection
pip install -r requirements.txt
```

### 2. 启动系统 Web 界面
切换至 `3、训练脚本` 目录，执行以下命令：
```bash
python web_test.py
```
或直接使用 Streamlit 命令行启动：
```bash
streamlit run web_test.py
```
启动后在浏览器打开 `http://localhost:8501` 即可开始使用。

