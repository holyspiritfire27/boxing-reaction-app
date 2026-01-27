import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# 邏輯核心類別 - 超高感度與回饋條版
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.state = 'WAIT_GUARD' 
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        self.command_display_until = 0
        
        # 數據與紀錄
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.record_max_speed = 0.0
        self.reaction_times_list = []
        
        # 運動學暫存
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心門檻調優 (極致敏感) ===
        self.EXTENSION_THRESHOLD = 0.035     # 手腕遠離肩膀 (原 0.04)
        self.ELBOW_LIFT_THRESHOLD = 0.65     # 手肘上升權重 (原 0.55，數值越大越靈敏)
        self.ARM_ANGLE_THRESHOLD = 85        # 手臂張開角度 (原 90)
        self.RETRACTION_THRESHOLD = 0.20     # 歸位判定 (放寬，讓程式更容易進入 READY)

        # 實時回饋值
        self.current_intensity = 0.0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def draw_feedback_bar(self, image, h, w):
        """ 右下角出拳幅度回饋條 """
        bar_w, bar_h = 200, 20
        start_x, start_y = w - 220, h - 40
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        # 填充 (根據強度改變顏色：綠->黃->紅)
        fill_w = int(self.current_intensity * bar_w)
        color = (0, 255, 0) if self.current_intensity < 0.7 else (0, 255, 255)
        if self.current_intensity >= 1.0: color = (0, 0, 255)
        
        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), color, -1)
        cv2.putText(image, "PUNCH INTENSITY", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 210), (340, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'PRE_START': ("READY...", (0, 255, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT_PENDING': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0))
        }
        status_text, color = status_map.get(self.state, ("IDLE", (255,255,255)))
        cv2.putText(image, status_text, (20, h - 175), font, 0.7, color, 2)

        r_time = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        v_speed = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        avg_r = sum(self.reaction_times_list) / len(self.reaction_times_list) if self.reaction_times_list else 0
        
        cv2.putText(image, f"Last Time: {r_time}", (20, h - 140), font, 0.7, (255,255,255), 2)
        cv2.putText(image, f"Last Speed: {v_speed}", (20, h - 110), font, 0.7, (255,255,255), 2)
        cv2.line(image, (20, h - 95), (320, h - 95), (100, 100, 100), 1)
        cv2.putText(image, f"Max Speed: {self.record_max_speed:.1f} m/s", (20, h - 65), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Avg React: {int(avg_r)} ms", (20, h - 35), font, 0.7, (0, 255, 255), 2)

    def process(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        current_time = time.time()
        
        self.draw_dashboard(image, h, w)
        self.draw_feedback_bar(image, h, w)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            # 重要：手肘相對於肩膀的高度指標 (Y 越小越高)
            # 我們計算：肩膀 Y - 手肘 Y。如果手肘高過肩膀，這值是正的且很大。
            l_elbow_height = l_sh.y - l_el.y 
            r_elbow_height = r_sh.y - r_el.y

            dist_l, dist_r = abs(l_wr.x - l_sh.x), abs(r_wr.x - r_sh.x)
            angle_l, angle_r = self.calculate_angle(l_sh, l_el, l_wr), self.calculate_angle(r_sh, r_el, r_wr)

            # 速度計算
            sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist if sh_dist > 0 else 0
            curr_v = 0
            if self.prev_landmarks:
                dt = current_time - self.prev_time
                if dt > 0:
                    l_v = (np.sqrt((l_wr.x - self.prev_landmarks[15].x)**2 + (l_wr.y - self.prev_landmarks[15].y)**2) * scale) / dt
                    r_v = (np.sqrt((r_wr.x - self.prev_landmarks[16].x)**2 + (r_wr.y - self.prev_landmarks[16].y)**2) * scale) / dt
                    curr_v = max(l_v, r_v)
            self.prev_landmarks = landmarks
            self.prev_time = current_time

            # 計算當前強度 (用於回饋條，0.0~1.0)
            target_dist = dist_l if self.target == 'LEFT' else dist_r
            target_angle = angle_l if self.target == 'LEFT' else angle_r
            target_h = l_elbow_height if self.target == 'LEFT' else r_elbow_height
            
            # 標準化強度 (取位移、角度、高度中最高的貢獻度)
            s_dist = min(1.0, target_dist / self.EXTENSION_THRESHOLD)
            s_angle = min(1.0, (target_angle - 40) / (self.ARM_ANGLE_THRESHOLD - 40))
            s_height = min(1.0, (target_h + 0.1) / (0.1 + 0.05)) # 讓手肘往上移動約 0.05 單位即滿
            self.current_intensity = max(s_dist, s_angle, s_height)

            # --- 狀態機邏輯 ---
            if self.state == 'WAIT_GUARD':
                self.current_intensity = 0
                if (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD):
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "BRING HANDS BACK", (int(w/2)-180, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.0
                    self.max_v_temp = 0.0

            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

            if self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                self.max_v_temp = max(self.max_v_temp, curr_v)

                # 判定邏輯 (三個條件滿足其一即中)
                # 1. 手肘高度上升 (相對於肩膀 Y 軸提升)
                hit_h = target_h > -0.05 # 手肘高度接近或超過肩膀
                # 2. 手臂打直角度
                hit_a = target_angle > self.ARM_ANGLE_THRESHOLD
                # 3. 手腕位移
                hit_d = target_dist > self.EXTENSION_THRESHOLD

                if hit_h or hit_a or hit_d:
                    self.last_reaction_time = elapsed * 1000
                    self.last_velocity = self.max_v_temp
                    self.reaction_times_list.append(self.last_reaction_time)
                    self.record_max_speed = max(self.record_max_speed, self.last_velocity)
                    self.state = 'RESULT_PENDING'
                    self.wait_until = self.command_display_until

                if elapsed > 3.0: self.state = 'WAIT_GUARD'

            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state, self.wait_until = 'RESULT', current_time + 2.0

            elif self.state == 'RESULT':
                if current_time > self.wait_until: self.state = 'WAIT_GUARD'

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        try:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except: return frame

def main():
    st.set_page_config(page_title="拳擊反應訓練 v14", layout="wide")
    st.title("🥊 AI 拳擊反應 - 敏感度增強版")
    webrtc_streamer(key="boxing-v14", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
