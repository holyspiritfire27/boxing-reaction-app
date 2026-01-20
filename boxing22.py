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
        self.state = 'WAIT_GUARD' # 初始狀態
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        
        # 數據記錄
        self.last_reaction_time = 0.0 # ms
        self.last_velocity = 0.0      # m/s
        self.last_hand = "None"
        
        # 速度計算變數
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # 判定參數
        self.current_extension = 0.0
        self.EXTENSION_THRESHOLD = 0.12     # 出拳判定門檻 (伸多長算打中)
        self.RETRACTION_THRESHOLD = 0.15    # 歸位判定門檻 (縮多短算歸位)
        self.MAX_EXTENSION_DISPLAY = 0.3    # 進度條顯示比例用

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dx = landmark.x - prev_landmark.x
        dy = landmark.y - prev_landmark.y
        dz = landmark.z - prev_landmark.z
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        return (dist_px * scale) / dt

    def draw_dashboard(self, image, h, w):
        """ 繪製儀表板 """
        # 半透明黑底
        overlay = image.copy()
        top_y = max(0, h - 160)
        cv2.rectangle(overlay, (10, top_y), (300, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        
        # --- 狀態顯示 ---
        if self.state == 'WAIT_GUARD': 
            status_text = "RESET: HANDS BACK" # 提示把手收回來
            status_color = (0, 165, 255) # 橘色
        elif self.state == 'PRE_START': 
            status_text = "READY..."
            status_color = (0, 255, 255) # 黃色
        elif self.state == 'STIMULUS': 
            status_text = "GO !!!"
            status_color = (0, 0, 255) # 紅色
        else:
            status_text = "RESULT"
            status_color = (0, 255, 0)
            
        cv2.putText(image, f"{status_text}", (20, h - 120), font, 0.8, status_color, 2)

        # --- 數據顯示 ---
        if self.last_reaction_time > 0:
            r_time_str = f"{int(self.last_reaction_time)} ms" 
        else:
            r_time_str = "---"
            
        vel_str = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time_str}", (20, h - 80), font, 0.9, white, 2)
        cv2.putText(image, f"Speed: {vel_str}", (20, h - 40), font, 0.8, white, 2)

        # --- Extension Check Bar (右下角) ---
        # 這是為了讓您確認系統有偵測到您的動作
        bar_x = 320
        bar_w = 200
        bar_h = 20
        bar_y = h - 40
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 2)
        
        # 紅色閾值線
        threshold_ratio = self.EXTENSION_THRESHOLD / self.MAX_EXTENSION_DISPLAY
        threshold_x = int(bar_x + threshold_ratio * bar_w)
        cv2.line(image, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_h + 5), (0,0,255), 2)
        
        # 填充
        fill_ratio = self.current_extension / self.MAX_EXTENSION_DISPLAY
        fill_len = int(fill_ratio * bar_w)
        fill_len = max(0, min(fill_len, bar_w))
        
        # 顏色邏輯: 超過閾值變綠
        color = (0, 255, 0) if self.current_extension > self.EXTENSION_THRESHOLD else (0, 255, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_len, bar_y + bar_h), color, -1)
        cv2.putText(image, "Reach", (bar_x, bar_y - 10), font, 0.5, white, 1)

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
            
            # 隨時計算伸展程度 (絕對值)
            dist_l = abs(left_wrist.x - left_shoulder.x)
            dist_r = abs(right_wrist.x - right_shoulder.x)
            self.current_extension = max(dist_l, dist_r)

            # 計算速度
            shoulder_dist = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                    (left_shoulder.y - right_shoulder.y)**2)
            scale_factor = self.SHOULDER_WIDTH_M / shoulder_dist if shoulder_dist > 0 else 0

            left_v = 0
            right_v = 0
            if self.prev_landmarks:
                left_v = self.calculate_velocity(left_wrist, self.prev_landmarks[15], scale_factor, dt)
                right_v = self.calculate_velocity(right_wrist, self.prev_landmarks[16], scale_factor, dt)
            
            self.prev_landmarks = landmarks

            # ==========================
            # 狀態機 (State Machine)
            # ==========================
            
            # 1. 歸位與準備 (WAIT_GUARD)
            if self.state == 'WAIT_GUARD':
                # 判定 A: 手舉高 (Y軸) - 寬鬆判定
                guard_height = 0.2
                is_hands_up = (left_wrist.y < left_shoulder.y + guard_height) and \
                              (right_wrist.y < right_shoulder.y + guard_height)
                
                # 判定 B: 手收回 (X軸) - 重要修正！
                # 手腕 X 必須很靠近肩膀 X，才算歸位。防止手伸著就開始下一局。
                is_retracted = (dist_l < self.RETRACTION_THRESHOLD) and \
                               (dist_r < self.RETRACTION_THRESHOLD)
                
                if is_hands_up and is_retracted:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                else:
                    # 提示畫面
                    msg = "NEXT ROUND"
                    sub_msg = "HANDS UP & BACK" # 提示要把手收回來
                    
                    # 閃爍效果
                    if int(current_time * 2) % 2 == 0:
                        cv2.putText(image, msg, (int(w/2)-150, int(h/2)-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(image, sub_msg, (int(w/2)-200, int(h/2)+50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # 2. 隨機等待
            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time

            # 3. 出題 (刺激)
            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                
                # 指令顯示 (加大字體)
                if elapsed < 0.8:
                    text = self.target + "!"
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    font_scale = 4
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 8)

                # 超時 (延長到 3 秒)
                if elapsed > 3.0:
                    self.state = 'WAIT_GUARD'

                # 擊打判定
                hit = False
                hit_v = 0
                
                if self.target == 'LEFT':
                    # 判斷: X軸伸展超過門檻
                    if (left_wrist.x < left_shoulder.x - self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = left_v
                else:
                    if (right_wrist.x > right_shoulder.x + self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = right_v
                
                if hit:
                    # 更新數據
                    self.last_reaction_time = elapsed * 1000 # ms
                    self.last_velocity = hit_v
                    self.last_hand = self.target
                    
                    # 狀態切換
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0 

            # 4. 顯示結果
            elif self.state == 'RESULT':
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

# ==========================================
# 串流與主程式
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

def main():
    st.set_page_config(page_title="拳擊反應訓練 v6", layout="wide")
    
    st.sidebar.title("🥊 拳擊反應 v6.0")
    st.sidebar.info(
        """
        **數據說明:**
        - **Time (ms)**: 反應速度，越低越好。
        - **Speed (m/s)**: 出拳速度。

        **重要技巧:**
        打完一拳後，請務必將**雙手收回下巴/肩膀附近**。
        如果畫面顯示 **RESET: HANDS BACK**，代表系統在等您收拳歸位。
        歸位後，下一局才會自動開始。
        """
    )
    
    st.title("🥊 AI 拳擊反應測試 (優化版)")
    st.markdown("請站在距離鏡頭約 2 公尺處，讓全身或上半身完整入鏡。")

    webrtc_streamer(
        key="boxing-reaction-v6",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
