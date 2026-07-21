# -*- coding: utf-8 -*-
# ====================================================
# 1. 系统底层初始化：拦截物理证书异常（全局置顶配置）
# ====================================================
import os
import pathlib
import ssl
import sys
import time
import warnings

ssl.SSLContext._load_windows_store_certs = lambda self, storename, purpose: None

# 抑制未来不兼容版本警告提示
warnings.filterwarnings("ignore", category=FutureWarning)

# 跨版本序列化兼容性补丁
try:
    sys.modules["pathlib._local"] = pathlib
except Exception:
    pass

import cv2
import numpy as np
import torch
from PIL import Image, ImageGrab
import streamlit as st
from streamlit.web import cli as stcli

# ====================================================
# 2. 系统全局配置：网页架构声明与路径智能寻址
# ====================================================
PROJECT_TITLE = "基于YOLOv5的城市道路车辆类别与行人检测"

st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

# 获取脚本所在根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 智能匹配 "4、权重文件" 目录
MODEL_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "4、权重文件", "best.pt"),
    os.path.join(PROJECT_ROOT, "4、权重文件", "weights", "best.pt"),
    r"G:\桌面\深度学习项目\第六组-基于YOLOv5的城市道路车辆类别与行人检测\项目源代码\4、权重文件\weights\best.pt",
    r"G:\桌面\深度学习项目\第六组-基于YOLOv5的城市道路车辆类别与行人检测\项目源代码\4、权重文件\best.pt",
]
MODEL_PATH = next(
    (p for p in MODEL_PATH_CANDIDATES if os.path.exists(p)),
    MODEL_PATH_CANDIDATES[0],
)

# YOLOv5 源码宿主路径
YOLO_SRC_CANDIDATES = [
    r"E:\pythonDemo\yolo5\yolov5-master",
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "yolov5-master")),
]
YOLO_SRC = next(
    (p for p in YOLO_SRC_CANDIDATES if os.path.exists(p)), YOLO_SRC_CANDIDATES[0]
)

init_grab = ImageGrab.grab()
max_w, max_h = init_grab.size


@st.cache_resource
def load_my_model():
    model = torch.hub.load(YOLO_SRC, "custom", path=MODEL_PATH, source="local")
    model.conf = 0.4
    model.augment = True
    return model


try:
    model = load_my_model()
except Exception as e:
    st.sidebar.error(
        f"❌ 模型加载失败，请检查权重文件路径是否正确！\n具体路径: {MODEL_PATH}\n报错详情: {e}"
    )

# ====================================================
# 3. 前端交互界面布局管理（使用持久化导航栏）
# ====================================================
st.title(f"🚦 {PROJECT_TITLE} 平台 v2.0")

# 🌟 使用带 Key 记忆的横向导航，彻底解决跳转 Tab1 和底部重复渲染问题
selected_tab = st.radio(
    "功能导航",
    ["📸 图片/文件夹识别", "🎬 视频离线检测", "🖥️ 屏幕任意区域实时透视"],
    horizontal=True,
    key="nav_selected_tab",
    label_visibility="collapsed",
)
st.markdown("---")

# ----------------------------------------------------
# 交互子模块一：静态图像/文件夹特征提取与批量推理
# ----------------------------------------------------
if selected_tab == "📸 图片/文件夹识别":
    st.header("图片多目标特征检测")
    uploaded_files = st.file_uploader(
        "请拖拽上传一张或多张交通路口图片：",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"已成功接收 {len(uploaded_files)} 张待处理数据。")
        if st.button("🚀 开始网页端批量推理"):
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                input_img = Image.open(file)
                results = model(input_img)
                annotated_img = results.render()[0]

                with cols[idx % 2]:
                    st.image(
                        annotated_img,
                        caption=f"成果图: {file.name}",
                        use_container_width=True,
                    )

# ----------------------------------------------------
# 交互子模块二：离线视频全量特征流检测
# ----------------------------------------------------
elif selected_tab == "🎬 视频离线检测":
    st.header("视频全量流检测")
    uploaded_file = st.file_uploader(
        "请上传一段交通监控短视频：", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:
        temp_video_path = "temp_user_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("视频上传解析完毕，已就绪。")

        if st.button("🎬 执行网页视频逐帧渲染"):
            video_cap = cv2.VideoCapture(temp_video_path)
            v_col1, v_col2, v_col3 = st.columns([1, 4, 1])
            with v_col2:
                video_frame_placeholder = st.empty()

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                annotated_frame = results.render()[0]

                video_frame_placeholder.image(
                    annotated_frame, channels="RGB", use_container_width=True
                )
                time.sleep(0.01)

            video_cap.release()
            st.balloons()
            st.success("🎉 该视频全量交通特征流提取完毕！")

# ----------------------------------------------------
# 交互子模块三：自定义屏幕物理区域捕获与实时动态同步透视
# ----------------------------------------------------
elif selected_tab == "🖥️ 屏幕任意区域实时透视":
    st.header("自定义屏幕区域动态追踪")

    st.markdown(
        """
    **测试视频链接：**
    - 🚗 [车辆检测测试视频](https://www.bilibili.com/video/BV1uS4y1v7qN/)
    - 🚶 [行人检测测试视频](https://www.bilibili.com/video/BV1fE411w7ac/)
    - 🚥 [交通路口综合监控视频](https://www.bilibili.com/video/BV1BWxse9E4T/)
    """
    )

    st.info(
        "💡 **操作说明**：点击下方按钮后，屏幕会自动弹窗置顶，用鼠标框选画面并按【空格键】确认，系统将**自动直接启动**实时检测流！"
    )

    if "roi_coords" not in st.session_state:
        st.session_state.roi_coords = None
    if "live_active" not in st.session_state:
        st.session_state.live_active = False

    col_btn1, col_btn2 = st.columns([2, 2])

    with col_btn1:
        if st.button("🎯 点击激活鼠标初次框选器"):
            full_screen = ImageGrab.grab()
            full_screen_bgr = cv2.cvtColor(
                np.array(full_screen), cv2.COLOR_RGB2BGR
            )

            roi_win_name = "Drag a box on your video and press Space"
            cv2.namedWindow(roi_win_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                roi_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.setWindowProperty(roi_win_name, cv2.WND_PROP_TOPMOST, 1)

            roi = cv2.selectROI(
                roi_win_name,
                full_screen_bgr,
                fromCenter=False,
                showCrosshair=True,
            )
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            if roi[2] > 0 and roi[3] > 0:
                st.session_state.roi_coords = roi
                st.session_state.live_active = True
                st.rerun()

    with col_btn2:
        if st.session_state.live_active:
            if st.button("🔴 停止实时检测"):
                st.session_state.live_active = False
                st.session_state.roi_coords = None
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                st.rerun()

    if (
        st.session_state.roi_coords
        and sum(st.session_state.roi_coords) > 0
        and st.session_state.live_active
    ):
        x, y, w, h = st.session_state.roi_coords

        if w >= h:
            c_col1, c_col2, c_col3 = st.columns([1, 5, 1])
        else:
            c_col1, c_col2, c_col3 = st.columns([2, 3, 2])

        with c_col2:
            live_placeholder = st.empty()

        ctrl_win = "Control Panel (Press R to Re-draw box)"
        cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(ctrl_win, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(ctrl_win, 450, 220)

        def nothing(v):
            pass

        cv2.createTrackbar("X_pos", ctrl_win, x, max_w, nothing)
        cv2.createTrackbar("Y_pos", ctrl_win, y, max_h, nothing)
        cv2.createTrackbar("Width", ctrl_win, w, max_w, nothing)
        cv2.createTrackbar("Height", ctrl_win, h, max_h, nothing)

        try:
            while st.session_state.live_active:
                x = cv2.getTrackbarPos("X_pos", ctrl_win)
                y = cv2.getTrackbarPos("Y_pos", ctrl_win)
                w = cv2.getTrackbarPos("Width", ctrl_win)
                h = cv2.getTrackbarPos("Height", ctrl_win)

                if w < 16:
                    w = 16
                if h < 16:
                    h = 16
                if x + w > max_w:
                    w = max_w - x
                if y + h > max_h:
                    h = max_h - y

                crop_img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                results = model(crop_img)
                annotated_img = results.render()[0]

                target_width = 750 if w >= h else 420
                scale_ratio = target_width / float(w)
                target_height = int(h * scale_ratio)
                img_resized = cv2.resize(
                    annotated_img, (target_width, target_height)
                )

                live_placeholder.image(
                    img_resized, channels="RGB", use_container_width=True
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("r") or key == ord("R"):
                    try:
                        cv2.destroyWindow(ctrl_win)
                    except Exception:
                        pass

                    full_screen = ImageGrab.grab()
                    full_screen_bgr = cv2.cvtColor(
                        np.array(full_screen), cv2.COLOR_RGB2BGR
                    )
                    re_win = "Re-draw Box (Space to confirm)"
                    cv2.namedWindow(re_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(
                        re_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                    cv2.setWindowProperty(re_win, cv2.WND_PROP_TOPMOST, 1)
                    new_roi = cv2.selectROI(
                        re_win,
                        full_screen_bgr,
                        fromCenter=False,
                        showCrosshair=True,
                    )
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass

                    if new_roi[2] > 0 and new_roi[3] > 0:
                        x, y, w, h = new_roi
                        st.session_state.roi_coords = new_roi

                    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(ctrl_win, cv2.WND_PROP_TOPMOST, 1)
                    cv2.resizeWindow(ctrl_win, 450, 220)
                    cv2.createTrackbar("X_pos", ctrl_win, x, max_w, nothing)
                    cv2.createTrackbar("Y_pos", ctrl_win, y, max_h, nothing)
                    cv2.createTrackbar("Width", ctrl_win, w, max_w, nothing)
                    cv2.createTrackbar("Height", ctrl_win, h, max_h, nothing)

                time.sleep(0.005)
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

# ====================================================
# 4. 系统运行时防重入控制网关
# ====================================================
if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())