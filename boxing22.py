import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp # <--- 標準匯入

# ==========================================
# 拳擊分析邏輯
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 0.9.0.1 支援這種標準寫法
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0 # Cloud 環境建議用 0 或 1 以提升效能
        )
        
        # 遊戲狀態
        self.counter = 0
        self.last_action_time = 0
        self.reaction_times = []
        self.target = None
        self.waiting_for_action = False
        self.start_time = 0

    def process(self, image):
        # 影像處理標準流程
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image_rgb)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # --- 遊戲核心邏輯 ---
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            current_time = time.time()
            
            # 出題
            if not self.target and (current_time - self.last_action_time > 2.0):
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.start_time = current_time
                self.waiting_for_action = True

            # 顯示指令
            if self.target:
                text = f"PUNCH {self.target}!"
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)

            # 偵測動作
            action_detected = None
            if left_wrist.x < left_elbow.x - 0.05: action_detected = 'LEFT'
            if right_wrist.x > right_elbow.x + 0.05: action_detected = 'RIGHT'

            # 判定得分
            if self.waiting_for_action and action_detected == self.target:
                reaction_time = current_time - self.start_time
                self.reaction_times.append(reaction_time)
                self.last_action_time = current_time
                self.target = None
                self.waiting_for_action = False
                self.counter += 1

            # 顯示分數
            cv2.rectangle(image, (0,0), (250, 80), (245,117,16), -1)
            cv2.putText(image, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            if self.reaction_times:
                avg_time = np.mean(self.reaction_times)
                cv2.putText(image, f'{avg_time:.2f}s', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return image

# ==========================================
# 影像處理器
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 鏡像
        img = self.logic.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Main
# ==========================================
def main():
    st.set_page_config(page_title="Boxing Reaction", layout="wide")
    st.title("🥊 Boxing Reaction Trainer")
    
    st.warning("正在載入 AI 模型，請稍候...")
    
    webrtc_streamer(
        key="boxing-reaction",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
