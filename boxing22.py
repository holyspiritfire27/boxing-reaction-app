import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp  # <--- 回歸標準寫法

# ==========================================
# 拳擊分析邏輯 (Logic Class)
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # 因為環境已經修復，我們使用標準的 MediaPipe 呼叫方式
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        self.target = None  # 'LEFT' or 'RIGHT'
        self.waiting_for_action = False
        self.start_time = 0

    def process(self, image):
        # 轉換顏色空間 BGR -> RGB
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 進行偵測
        results = self.pose.process(image_rgb)
        
        # 畫回原本的圖上
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 取得畫面尺寸
        h, w, c = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # -------------------------------------------------------
            # 遊戲邏輯
            # -------------------------------------------------------
            
            # 取得左手座標
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            
            # 取得右手座標
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            # 隨機出題
            current_time = time.time()
            
            # 如果目前沒有目標，每隔幾秒生成一個新目標
            if not self.target and (current_time - self.last_action_time > 3):
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.start_time = current_time
                self.waiting_for_action = True

            # 顯示指令
            if self.target:
                color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                text = f"PUNCH {self.target}!"
                # 文字外框(黑色)以增加對比度
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 8, cv2.LINE_AA)
                # 文字本體
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

            action_detected = None
            
            # 簡單判斷：手腕比手肘更遠離身體中心 (X軸判斷)
            # 左手向左伸展
            if left_wrist.x < left_elbow.x - 0.05: 
                action_detected = 'LEFT'
            
            # 右手向右伸展
            if right_wrist.x > right_elbow.x + 0.05:
                action_detected = 'RIGHT'

            # 檢查是否擊中目標
            if self.waiting_for_action and action_detected == self.target:
                reaction_time = current_time - self.start_time
                self.reaction_times.append(reaction_time)
                self.last_action_time = current_time
                self.target = None # 重置
                self.waiting_for_action = False
                self.counter += 1

            # 顯示狀態
            cv2.rectangle(image, (0,0), (250, 73), (245,117,16), -1)
            cv2.putText(image, 'HITS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter), (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            if self.reaction_times:
                avg_time = np.mean(self.reaction_times)
                cv2.putText(image, f'Avg: {avg_time:.2f}s', (260, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return image

# ==========================================
# WebRTC 影像處理器
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 翻轉影像 (鏡像效果)，讓操作更直覺
        img = cv2.flip(img, 1)
        
        # 交給邏輯層處理
        img = self.logic.process(img)
        
        # 再翻轉回來嗎？通常不需要，因為 webrtc 會直接顯示處理後的
        # 但要注意左右手判斷邏輯是否受翻轉影響
        # 這裡為了簡單，我們在 process 內部處理的是鏡像後的圖
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Streamlit 主程式
# ==========================================
def main():
    st.set_page_config(page_title="Boxing Reaction App", layout="wide")
    
    st.title("🥊 Boxing Reaction Trainer")
    st.write("請允許瀏覽器存取攝影機。如果是第一次執行，可能需要等待幾秒鐘載入模型。")

    # 啟動 WebRTC
    webrtc_streamer(
        key="boxing",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
