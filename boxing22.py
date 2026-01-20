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
        self.round_count = 1 # 新增回合計數器
        
        # 數據記錄
        self.last_reaction_time = 0.0 # ms
        self.last_velocity = 0.0      # m/s
        self.last_hand = "None"
        
        # 速度計算
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # Debug & 閾值
        self.current_extension = 0.0
        self.EXTENSION_THRESHOLD = 0.12 # 出拳判定門檻
        self.MAX_EXTENSION_DISPLAY = 0.3

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
        top_y = max(0, h - 200) # 稍微加高一點以容納 Round
        cv2.rectangle(overlay, (10, top_y), (300, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        
        # --- 狀態顯示 ---
        status_text = "READY"
        status_color = white
        
        if self.state == 'WAIT_GUARD': 
            status_text = "RESET (HANDS UP)"
            status_color = (0, 165, 255) # 橘色提示等待重置
        elif self.state == 'PRE_START': 
            status_text = "WAIT..."
            status_color = yellow
        elif self.state == 'STIMULUS': 
            status_text = "PUNCH !!!"
            status_color = (0, 0, 255) # 紅色
        elif self.state == 'RESULT': 
            status_text = "RESULT"
            status_color = (0, 255, 0) # 綠色
        
        cv2.putText(image, f"ROUND: {self.round_count}", (20, h - 170), font, 0.8, yellow, 2)
        cv2.putText(image, f"{status_text}", (20, h - 135), font, 0.7, status_color, 2)

        # --- 數據顯示 ---
        if self.last_reaction_time > 0:
            r_time_str = f"{int(self.last_reaction_time)} ms" 
        else:
            r_time_str = "---"
            
        vel_str = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time_str}", (20, h - 95), font, 0.9, white, 2)
        cv2.putText(image, f"Speed: {vel_str}", (20, h - 55), font, 0.8, white, 2)
        cv2.putText(image, f"Last: {self.last_hand}", (20, h - 20), font, 0.7, (200, 200, 200), 1)

        # --- Extension Check Bar (右下角) ---
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
        
        color = (0, 255, 0) if self.current_extension > self.EXTENSION_THRESHOLD else (0, 255, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_len, bar_y + bar_h), color, -1)
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
            
            # 更新伸展程度 (用於 Bar)
            dist_l = abs(left_wrist.x - left_shoulder.x)
            dist_r = abs(right_wrist.x - right_shoulder.x)
            self.current_extension = max(dist_l, dist_r)

            # 計算比例與速度
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
            
            # 1. 等待護臉 (重置狀態)
            if self.state == 'WAIT_GUARD':
                # 判定邏輯放寬：手腕的高度只要在「肩膀下方一點點」也算 (y座標 + 0.15)
                # 這樣不用舉太高就能觸發，體驗較好
                guard_threshold = 0.15 
                
                is_left_up = left_wrist.y < (left_shoulder.y + guard_threshold)
                is_right_up = right_wrist.y < (right_shoulder.y + guard_threshold)
                
                if is_left_up and is_right_up:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                    self.round_count += 1 # 回合 +1
                else:
                    # 螢幕中央大字提示重置
                    cv2.putText(image, "NEXT ROUND", (int(w/2)-150, int(h/2)-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(image, "HANDS UP!", (int(w/2)-130, int(h/2)+40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # 2. 隨機等待
            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time

            # 3. 出題 (刺激)
            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                
                # 指令顯示 0.6 秒
                if elapsed < 0.6:
                    text = self.target + "!"
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    font_scale = 3
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 5)

                # 超時重來
                if elapsed > 2.0:
                    self.state = 'WAIT_GUARD'

                # 擊打判定
                hit = False
                hit_v = 0
                
                if self.target == 'LEFT':
                    if (left_wrist.x < left_shoulder.x - self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = left_v
                else:
                    if (right_wrist.x > right_shoulder.x + self.EXTENSION_THRESHOLD):
                        hit = True
                        hit_v = right_v
                
                if hit:
                    self.last_reaction_time = elapsed * 1000 # 轉為 ms
                    self.last_velocity = hit_v
                    self.last_hand = self.target
                    self.state = 'RESULT'
                    # 結果顯示 2.5 秒後進入下一輪等待
                    self.wait_until = current_time + 2.5 

            # 4. 顯示結果
            elif self.state == 'RESULT':
                # 這裡只負責等待，數據顯示交給 draw_dashboard
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
    st.set_page_config(page_title="拳擊反應訓練 v5", layout="wide")
    
    st.sidebar.title("🥊 拳擊反應 v5.0")
    st.sidebar.info(
        """
        **操作流程:**
        1. **Start**: 舉起雙手 (Hands Up) 啟動。
        2. **Punch**: 看到指令快速出拳。
        3. **Reset**: 出完拳後，**請再次舉起雙手** 來啟動下一回合！
        
        **儀表板:**
        - **Round**: 目前回合數 (確定程式在跑)。
        - **State**: 目前狀態 (Reset = 等待您舉手)。
        - **Time**: 反應速度 (ms)。
        """
    )
    
    st.title("🥊 AI 拳擊反應測試 (流暢版)")
    st.markdown("如果數據沒更新，請觀察畫面是否顯示 **RESET (HANDS UP)**，若是，請將雙手舉回下巴高度即可。")

    webrtc_streamer(
        key="boxing-reaction-v5",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
