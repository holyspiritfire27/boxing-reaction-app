import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# 邏輯核心類別
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 設定
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 為了效能，Complexity 設為 0 或 1
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # --- 狀態機變數 ---
        # 狀態列表: 'IDLE' (閒置), 'WAIT_GUARD' (等待護臉), 'PRE_START' (隨機等待), 'STIMULUS' (出題), 'RESULT' (結果)
        self.state = 'WAIT_GUARD'
        self.target = None          # 'LEFT' or 'RIGHT'
        self.start_time = 0         # 出題的時間點
        self.stimulus_duration = 0.5 # 指令只顯示 0.5 秒
        self.wait_until = 0         # 用於計時器
        
        # --- 數據記錄 ---
        self.last_reaction_time = 0
        self.last_velocity = 0.0
        self.last_hand = ""
        
        # --- 速度計算變數 ---
        self.prev_landmarks = None
        self.prev_time = 0
        
        # 參數設定
        self.SHOULDER_WIDTH_M = 0.45  # 假設一般人肩寬約 45 公分 (用來推算真實速度)

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        """計算瞬時速度 (m/s)"""
        if dt <= 0: return 0
        # 計算位移 (歐幾里得距離)
        dx = landmark.x - prev_landmark.x
        dy = landmark.y - prev_landmark.y
        dz = landmark.z - prev_landmark.z
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 轉換成真實距離 (公尺) 並除以時間
        velocity = (dist_px * scale) / dt
        return velocity

    def process(self, image):
        # 1. 影像前處理
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 進行骨架偵測
        results = self.pose.process(image_rgb)
        
        # 3. 轉回 BGR 供繪圖
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 繪製骨架連線
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # --- 關鍵點獲取 ---
            # 鼻子(0), 左肩(11), 右肩(12), 左手腕(15), 右手腕(16)
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # --- 計算比例尺 (Pixels per Meter) ---
            # 計算雙肩在畫面中的距離
            shoulder_dist_normalized = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                               (left_shoulder.y - right_shoulder.y)**2)
            
            # 如果偵測不到肩膀，避免除以零
            scale_factor = 0
            if shoulder_dist_normalized > 0:
                # 比例尺：真實肩寬 (m) / 畫面肩寬 (normalized)
                scale_factor = self.SHOULDER_WIDTH_M / shoulder_dist_normalized

            # --- 計算手腕速度 ---
            left_v = 0
            right_v = 0
            if self.prev_landmarks:
                left_v = self.calculate_velocity(left_wrist, self.prev_landmarks[15], scale_factor, dt)
                right_v = self.calculate_velocity(right_wrist, self.prev_landmarks[16], scale_factor, dt)
            
            # 更新上一幀紀錄
            self.prev_landmarks = landmarks

            # =========================================================
            # 遊戲狀態機 (Game State Machine)
            # =========================================================
            
            # 狀態 1: 等待護臉 (Ready Stance)
            if self.state == 'WAIT_GUARD':
                # 提示文字 (為了防止亂碼，使用英文，但在下方UI顯示中文)
                cv2.putText(image, "HANDS UP!", (int(w*0.3), int(h*0.5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
                
                # 判定邏輯：雙手手腕高度 (y) 高於肩膀 (數值越小越搞) 且 接近鼻子
                # 簡單判定：只要手腕高於肩膀即可
                if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                    self.state = 'PRE_START'
                    # 隨機等待 1.5 ~ 3.5 秒
                    self.wait_until = current_time + random.uniform(1.5, 3.5)

            # 狀態 2: 隨機等待 (Pre-Start)
            elif self.state == 'PRE_START':
                # 顯示 ... 表示準備中
                cv2.putText(image, "...", (int(w*0.45), int(h*0.5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 4)
                
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time

            # 狀態 3: 出題與動作偵測 (Stimulus & Action)
            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                
                # 顯示指令：只顯示 0.5 秒
                if elapsed < 0.5:
                    text = "LEFT!" if self.target == 'LEFT' else "RIGHT!"
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    # 顯示在畫面正中央
                    font_scale = 3
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 5, cv2.LINE_AA)
                
                # 偵測出拳 (Timeout 設為 2秒，超時沒打就重來)
                if elapsed > 2.0:
                    self.state = 'WAIT_GUARD' # 超時，重來
                
                # 判定邏輯：手伸直 (X軸大幅超過肩膀)
                hit_detected = False
                hit_velocity = 0
                
                if self.target == 'LEFT':
                    # 左手向左伸 (畫面左邊 x 變小)
                    if left_wrist.x < left_shoulder.x - 0.2: 
                        hit_detected = True
                        hit_velocity = left_v
                else:
                    # 右手向右伸 (畫面右邊 x 變大)
                    if right_wrist.x > right_shoulder.x + 0.2: 
                        hit_detected = True
                        hit_velocity = right_v
                
                if hit_detected:
                    self.last_reaction_time = elapsed
                    self.last_velocity = hit_velocity
                    self.last_hand = self.target
                    self.state = 'RESULT'
                    self.wait_until = current_time + 3.0 # 結果顯示 3 秒

            # 狀態 4: 顯示結果 (Result)
            elif self.state == 'RESULT':
                # 將數據顯示在左下角 (Bottom-Left)，避免遮擋
                # 畫一個半透明黑底
                overlay = image.copy()
                cv2.rectangle(overlay, (10, h-150), (350, h-10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                # 顯示數據
                cv2.putText(image, f"Target: {self.last_hand}", (20, h-110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 反應時間 (綠色)
                cv2.putText(image, f"Time: {self.last_reaction_time:.3f} s", (20, h-70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 出拳速度 (黃色)
                cv2.putText(image, f"Speed: {self.last_velocity:.1f} m/s", (20, h-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

# ==========================================
# Video Processor
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            # 鏡像翻轉：讓使用者的左手對應畫面的左邊，更直覺
            img = cv2.flip(img, 1)
            img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

# ==========================================
# Main App
# ==========================================
def main():
    st.set_page_config(page_title="拳擊反應訓練", layout="wide")
    
    # 側邊欄說明 (中文化)
    st.sidebar.title("🥊 拳擊反應訓練")
    st.sidebar.info(
        """
        **操作指南:**
        1. **雙手護臉 (Hands Up)**: 將雙手舉至臉部高度以啟動遊戲。
        2. **等待指令**: 畫面會隨機顯示 "LEFT!" (左) 或 "RIGHT!" (右)。
        3. **快速出拳**: 看到指令後，以最快速度向對應方向出拳！
        
        **顯示數據:**
        - **Time**: 反應時間 (秒) - 越短越好
        - **Speed**: 出拳末端速度 (m/s) - 越快越好
        """
    )
    
    st.title("🥊 AI 拳擊反應與速度測試")
    st.markdown("請點擊下方 **Start** 按鈕開啟攝影機。請確保全身入鏡，並站在距離鏡頭約 1.5 ~ 2 公尺處。")

    webrtc_streamer(
        key="boxing-reaction",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
