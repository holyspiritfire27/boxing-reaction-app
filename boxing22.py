import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# 邏輯核心類別 - 介面回歸與循環優化版
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
        
        # 數據記錄
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.max_v_temp = 0.0
        self.trigger_reason = "" 
        
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 極致門檻設定 ===
        self.EXTENSION_THRESHOLD = 0.04      
        self.RETRACTION_THRESHOLD = 0.18     
        self.ELBOW_LIFT_THRESHOLD = 0.55     
        self.ARM_ANGLE_THRESHOLD = 90        

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dist_px = np.sqrt((landmark.x - prev_landmark.x)**2 + (landmark.y - prev_landmark.y)**2)
        return (dist_px * scale) / dt

    def draw_dashboard(self, image, h, w):
        """ 恢復左下角儀表板介面 """
        overlay = image.copy()
        # 繪製半透明黑框
        cv2.rectangle(overlay, (10, h - 160), (320, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        
        # 狀態顯示
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0)),
            'PRE_START': ("READY...", (0, 255, 255))
        }
        status_text, color = status_map.get(self.state, ("IDLE", white))
        cv2.putText(image, status_text, (20, h - 120), font, 0.8, color, 2)

        # 數據顯示
        r_time = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        v_speed = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time}", (20, h - 85), font, 0.9, white, 2)
        cv2.putText(image, f"Speed: {v_speed}", (20, h - 50), font, 0.8, white, 2)
        
        if self.state == 'RESULT':
            cv2.putText(image, f"By: {self.trigger_reason}", (20, h - 20), font, 0.5, (200, 200, 200), 1)

    def process(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        self.draw_dashboard(image, h, w)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            dist_l, dist_r = abs(l_wr.x - l_sh.x), abs(r_wr.x - r_sh.x)
            angle_l, angle_r = self.calculate_angle(l_sh, l_el, l_wr), self.calculate_angle(r_sh, r_el, r_wr)

            sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist if sh_dist > 0 else 0
            curr_v = 0
            if self.prev_landmarks:
                l_v = self.calculate_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                curr_v = max(l_v, r_v)
            self.prev_landmarks = landmarks

            # === 狀態機控制 ===
            if self.state == 'WAIT_GUARD':
                if (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD):
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "BRING HANDS BACK", (int(w/2)-180, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.max_v_temp = 0.0

            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                self.max_v_temp = max(self.max_v_temp, curr_v)

                # 指令停留 0.5 秒
                if elapsed <= 0.5:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

                # 判定邏輯 (隨時監測)
                t_dist = dist_l if self.target == 'LEFT' else dist_r
                t_angle = angle_l if self.target == 'LEFT' else angle_r
                t_el_y, t_sh_y = (l_el.y, l_sh.y) if self.target == 'LEFT' else (r_el.y, r_sh.y)

                hit, reason = False, ""
                if t_dist > self.EXTENSION_THRESHOLD: hit, reason = True, "Reach"
                elif t_angle > self.ARM_ANGLE_THRESHOLD: hit, reason = True, "Straight"
                elif t_el_y < (t_sh_y + self.ELBOW_LIFT_THRESHOLD): hit, reason = True, "Elbow Lift"

                if hit:
                    self.last_reaction_time = elapsed * 1000
                    self.last_velocity = self.max_v_temp
                    self.trigger_reason = reason
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0 # 結果呈現 2 秒

                if elapsed > 3.0: # 超時重置
                    self.state = 'WAIT_GUARD'

            elif self.state == 'RESULT':
                # 在結果階段持續顯示數據
                cv2.putText(image, "PERFECT!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")

def main():
    st.set_page_config(page_title="拳擊反應訓練 v11", layout="wide")
    st.title("🥊 AI 拳擊訓練 (指令 0.5s / 結果 2s)")
    webrtc_streamer(key="boxing-v11", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
