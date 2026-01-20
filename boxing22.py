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
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # 狀態管理
        self.state = 'WAIT_GUARD' # WAIT_GUARD, PRE_START, STIMULUS, RESULT
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        
        # 數據記錄
        self.last_reaction_time = 0.0 # 單位: ms
        self.last_velocity = 0.0      # 單位: m/s
        self.last_hand = "None"
        
        # 速度計算
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # Debug: 記錄目前的伸展程度
        self.current_extension = 0.0
        
        # 設定判定門檻 (越小越容易觸發)
        self.EXTENSION_THRESHOLD = 0.12 
        # 設定最大顯示範圍 (用於繪製進度條比例)
        self.MAX_EXTENSION_DISPLAY = 0.3

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dx = landmark.x - prev_landmark.x
        dy = landmark.y - prev_landmark.y
        dz = landmark.z - prev_landmark.z
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        return (dist_px * scale) / dt

    def draw_dashboard(self, image, h, w):
        """ 繪製常駐儀表板 """
        # 1. 左下角半透明黑底
        overlay = image.copy()
        top_y = max(0, h - 180)
        cv2.rectangle(overlay, (10, top_y), (300, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # 2. 顯示固定標籤
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        
        # 狀態顯示
        status_text = "READY"
        if self.state == 'WAIT_GUARD': status_text = "HANDS UP!"
        elif self.state == 'PRE_START': status_text = "WAIT..."
        elif self.state == 'STIMULUS': status_text = "PUNCH!"
        elif self.state == 'RESULT': status_text = "RESULT"
        
        cv2.putText(image, f"STATE: {status_text}", (20, h - 140), font, 0.7, (0, 255, 255), 2)

        # 數據顯示 (轉為 ms 整數顯示)
        if self.last_reaction_time > 0:
            r_time_str = f"{int(self.last_reaction_time)} ms" 
        else:
            r_time_str = "---"
            
        vel_str = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time_str}", (20, h - 100), font, 0.9, white, 2)
        cv2.putText(image, f"Speed: {vel_str}", (20, h - 60), font, 0.8, white, 2)
        cv2.putText(image, f"Last: {self.last_hand}", (20, h - 25), font, 0.7, (200, 200, 200), 1)

        # 3. 繪製伸展力度條 (Extension Check)
        bar_x = 320
        bar_w = 200
        bar_h = 20
        bar_y = h - 40
        
        # 外框
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 2)
        
        # 閾值線 (紅線) - 根據 0.12 計算位置
        threshold_ratio = self.EXTENSION_THRESHOLD / self.MAX_EXTENSION_DISPLAY
        threshold_x = int(bar_x + threshold_ratio * bar_w)
        
        # 畫紅線
        cv2.line(image, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_h + 5), (0,0,255), 2)
        
        # 填充條 (根據目前伸展程度)
        fill_ratio = self.current_extension / self.MAX_EXTENSION_DISPLAY
        fill_len = int(fill_ratio * bar_w)
        fill_len = max(0, min(fill_len, bar_w))
        
        # 顏色邏輯：超過閾值變綠色，否則黃色
        color = (0, 255, 0) if self.current_extension > self.EXTENSION_THRESHOLD else (0, 255, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_len, bar_y + bar_h), color, -1)
        
        # 文字標示
        cv2.putText(image, "Reach Check", (bar_x, bar_y - 10), font, 0.5, white, 1)

    def process(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        # 繪製儀表板
        self.draw_dashboard(image, h, w)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 畫骨架
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 獲取關鍵點
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # 計算目前的「橫向伸展距離」 (絕對值)
            # 這決定了下方的綠色條有多長
            dist_l = abs(left_wrist.x - left_shoulder.x)
            dist_r = abs(right_wrist.x - right_shoulder.x)
            self.current_extension = max(dist_l, dist_r)

            # 計算比例尺
            shoulder_dist = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                    (left_shoulder.y - right_shoulder.y)**2)
            scale_factor = self.SHOULDER_WIDTH_M / shoulder_dist if shoulder_dist > 0 else 0

            # 計算速度
            left_v = 0
            right_v = 0
            if self.prev_landmarks:
                left_v = self.calculate_velocity(left_wrist, self.prev_landmarks[15], scale_factor, dt)
                right_v = self.calculate_velocity(right_wrist, self.prev_landmarks[16], scale_factor, dt)
            
            self.prev_landmarks = landmarks

            # ==========================
            # 狀態機
            # ==========================
            
            # 1. 等待護臉
            if self.state == 'WAIT_GUARD':
                # 判斷標準：手腕高於肩膀 (Y座標比較小)
                is_guarding = (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)
                
                if is_guarding:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "RAISE HANDS", (int(w/2)-100, int(h/2)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 2. 隨機等待
            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time

            # 3. 出題
            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                
                # 顯示指令 (0.5秒)
                if elapsed < 0.5:
                    text = self.target + "!"
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    font_scale = 3
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 5)

                # 超時判定 (2秒沒打就重來)
                if elapsed > 2.0:
                    self.state = 'WAIT_GUARD'

                # 出拳判定 (門檻已降至 0.12)
                hit = False
                hit_v = 0
                
                if self.target == 'LEFT':
                    # 左手往左伸 (x 變小)
                    if (left_wrist.x < left_shoulder.x - self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = left_v
                else:
                    # 右手往右伸 (x 變大)
                    if (right_wrist.x > right_shoulder.x + self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = right_v
                
                if hit:
                    # 將秒轉換為毫秒 (ms)
                    self.last_reaction_time = elapsed * 1000 
                    self.last_velocity = hit_v
                    self.last_hand = self.target
                    self.state = 'RESULT'
                    self.wait_until = current_time + 3.0

            # 4. 顯示結果
            elif self.state == 'RESULT':
                # 數據已由 draw_dashboard 處理
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

# ==========================================
# 串流處理器
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1) # 鏡像
            img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Frame Error: {e}")
            return frame

def main():
    st.set_page_config(page_title="拳擊反應訓練", layout="wide")
    
    st.sidebar.title("🥊 拳擊反應 v4.0")
    st.sidebar.info(
        """
        **數據說明:**
        - **Time**: 反應時間，單位毫秒 (ms)。
          - 頂尖選手: 100-200 ms
          - 一般人: 250-300 ms
        - **Speed**: 瞬間出拳速度 (m/s)。
        
        **判定指示:**
        - 觀察畫面下方的 **Reach Check (綠色條)**。
        - 只要綠色條超過紅線 (門檻 0.12)，即判定擊中。
        """
    )
    
    st.title("🥊 AI 拳擊反應測試 (ms版)")
    st.markdown("請舉起雙手護臉 (Hands Up) 開始測試。")

    webrtc_streamer(
        key="boxing-reaction-v4",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
