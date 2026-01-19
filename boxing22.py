import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random

# ==========================================
# 核心修復部分：MediaPipe 引用方式
# ==========================================
# 在 Streamlit Cloud (Python 3.11/3.13) 上，直接呼叫 mp.solutions.pose 有時會失效
# 因此我們這裡使用 "from ... import ..." 的顯式寫法來繞過這個問題
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    st.error("無法匯入 MediaPipe，請確認 requirements.txt 包含 mediapipe 和 protobuf==3.20.3")

# ==========================================
# 拳擊分析邏輯 (Logic Class)
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # 使用上面顯式引用的模組，而不是 mp.solutions.pose
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        # 初始化 Pose 模型
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy (建議 1 平衡速度與準確度)
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
            # 這裡您可以放入您原本的偵測邏輯
            # 以下是一個簡單的範例：偵測出拳 (手腕超過手肘)
            # -------------------------------------------------------
            
            # 取得左手座標
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            
            # 取得右手座標
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            # 簡單的邏輯：隨機出題
            current_time = time.time()
            
            # 如果目前沒有目標，每隔幾秒生成一個新目標
            if not self.target and (current_time - self.last_action_time > 3):
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.start_time = current_time
                self.waiting_for_action = True

            # 顯示指令
            if self.target:
                color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                cv2.putText(image, f"PUNCH {self.target}!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

            # 偵測動作是否完成 (簡單判斷：手腕 X 軸大幅移動或 Y 軸高於鼻子等，這裡示範 X 軸伸展)
            # 注意：MediaPipe 座標是歸一化的 (0~1)
            
            action_detected = None
            
            # 簡單判斷：如果手腕非常接近相機 (z 軸) 或 手伸直
            # 這裡用一個簡單的視覺判斷：手腕比手肘更遠離身體中心
            # (這只是一個範例邏輯，請替換回您原本的判定代碼)
            
            # 假設：當左手腕的 x < 左手肘 x (畫面左邊) -> 左拳
            if left_wrist.x < left_elbow.x - 0.1:
                action_detected = 'LEFT'
            
            # 假設：當右手腕的 x > 右手肘 x (畫面右邊) -> 右拳
            if right_wrist.x > right_elbow.x + 0.1:
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
                cv2.putText(image, f'Avg Time: {avg_time:.2f}s', (260, 60), 
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
        
        # 交給邏輯層處理
        img = self.logic.process(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Streamlit 主程式
# ==========================================
def main():
    st.set_page_config(page_title="Boxing Reaction App", layout="wide")
    
    st.title("🥊 Boxing Reaction Trainer")
    st.write("這是一個使用 MediaPipe 的拳擊反應測試。請允許瀏覽器存取攝影機。")

    st.sidebar.title("設定")
    st.sidebar.info("請站在距離鏡頭約 1.5 ~ 2 公尺處，確保全身入鏡。")

    # 啟動 WebRTC
    webrtc_streamer(
        key="boxing",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
