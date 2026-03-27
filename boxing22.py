import os
import sys
import subprocess

# 🚨 暴力洗地補丁：在載入任何套件前，無情殺掉壞掉的標準版 OpenCV，並確保 headless 存活
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.74"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import streamlit as st
import av
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import io
from streamlit_webrtc import webrtc_streamer

# ==========================================
# 1. 頁面與 MediaPipe 初始化
# ==========================================
st.set_page_config(page_title="Boxing Reaction App", page_icon="🥊", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. 定義核心影像處理器 (完全無 cv2)
# ==========================================
class BoxingPoseProcessor:
    def __init__(self):
        # 初始化 MediaPipe Pose 模型
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 用來儲存最後一幀畫面，供截圖下載使用
        self.last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 👉 1. 直接取得 RGB 格式的 numpy array
        img = frame.to_ndarray(format="rgb24")

        # 👉 2. 進行骨架偵測
        results = self.pose.process(img)

        # 👉 3. 畫出骨架
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # 👉 4. 使用 PIL 畫 UI (標靶、文字)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # 畫一個虛擬打擊目標 (紅色框框)
        draw.rectangle([(50, 50), (150, 150)], outline=(255, 0, 0), width=3)
        # 寫上狀態文字
        draw.text((55, 55), "TARGET", fill=(255, 0, 0))

        # 將畫好的圖片轉回 numpy array
        img = np.array(img_pil)

        # 👉 5. 將結果存下來供截圖使用，並回傳給 WebRTC 顯示
        self.last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# ==========================================
# 3. Streamlit UI 與主程式邏輯
# ==========================================
def main():
    st.title("🥊 拳擊反應訓練 (無 cv2 極速穩定版)")
    st.markdown("這個版本專為 Streamlit Cloud 優化，已完全移除 `cv2` 依賴，享受更低延遲的體驗！")

    # 啟動 WebRTC 串流
    ctx = webrtc_streamer(
        key="boxing-reaction",
        video_processor_factory=BoxingPoseProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    # ==========================================
    # 4. 截圖下載功能 (免 cv2，使用 PIL + io)
    # ==========================================
    st.markdown("---")
    st.subheader("📸 成果截圖")
    
    # 確保攝影機有開啟且有抓到畫面
    if ctx.video_processor:
        if st.button("擷取當下畫面"):
            if ctx.video_processor.last_frame is not None:
                # 取得最後一幀並用 PIL 轉成 JPEG
                img_to_save = ctx.video_processor.last_frame
                image = Image.fromarray(img_to_save)
                
                # 存入記憶體緩衝區
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                
                # 顯示下載按鈕
                st.success("截圖成功！請點擊下方按鈕下載：")
                st.download_button(
                    label="⬇️ 下載圖片 (result.jpg)",
                    data=buf.getvalue(),
                    file_name="boxing_result.jpg",
                    mime="image/jpeg"
                )
            else:
                st.warning("請先對著鏡頭擺個姿勢，等畫面出來後再截圖喔！")
    else:
        st.info("請先點擊上方 'START' 開啟鏡頭。")

if __name__ == "__main__":
    main()
