import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random

# ==========================================
# 關鍵修正：強制顯式匯入 MediaPipe 模組
# ==========================================
# 不要使用 mp.solutions.pose，改用以下方式直接匯入：
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# ==========================================
# 拳擊分析邏輯
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # 使用上方強制匯入的變數
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        # 初始化 Pose 模型
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # 遊戲狀態變數
        self.stage = None
        self.counter = 0
        self.last_action_time = 0
        self.reaction_times = []
        self.target = None
        self.waiting_for_action = False
        self.start_time = 0

    def process(self, image):
        # 1. 影像前處理
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. MediaPipe 偵測
        results = self.pose.process(image_rgb)
        
        # 3. 轉回 BGR 以便繪圖
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # -------------------------------------------------------
            # 遊戲判定邏輯
            # -------------------------------------------------------
            
            # 取得關鍵點座標
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            # 時間管理
            current_time = time.time()
            
            # 生成新目標
            if not self.target and (current_time - self.last_action_time > 2.0):
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.start_time = current_time
                self.waiting_for_action = True

            # 畫面顯示指令
            if self.target:
                text = f"PUNCH {self.target}!"
                color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                # 黑色描邊
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 8, cv2.LINE_AA)
                # 彩色字體
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

            # 動作偵測 (簡單版：手腕 X 軸超過手肘)
            action_detected = None
            if left_wrist.x < left_elbow.x - 0.05:  # 畫面左側
                action_detected = 'LEFT'
            if right_wrist.x > right_elbow.x + 0.05: # 畫面右側
                action_detected = 'RIGHT'

            # 判定得分
            if self.waiting_for_action and action_detected == self.target:
                reaction_time = current_time - self.start_time
                self.reaction_times.append(reaction_time)
                self.last_action_time = current_time
                self.target = None
                self.waiting_for_action = False
                self.counter += 1

            # 顯示數據儀表板
            cv2.rectangle(image, (0,0), (300, 80), (245,117,16), -1)
            
            # 次數
            cv2.putText(image, 'HITS', (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # 平均反應時間
            if self.reaction_times:
                avg_time = np.mean(self.reaction_times)
                cv2.putText(image, f'Avg: {avg_time:.2f}s', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return image

# ==========================================
# 影像處理器 Class
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 鏡像翻轉 (讓使用者的左手對應畫面左邊)
        img = cv2.flip(img, 1)
        
        # 執行邏輯
        img = self.logic.process(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 主程式 Entry Point
# ==========================================
def main():
    st.set_page_config(page_title="Boxing Reaction", layout="wide")
    st.title("🥊 Boxing Reaction Trainer")
    
    st.write("如果是第一次執行，請等待約 10 秒鐘載入模型。")
    st.info("請點擊下方 Start，並允許瀏覽器使用攝影機。")

    webrtc_streamer(
        key="boxing-reaction",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
