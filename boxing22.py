import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp
import math

# ==========================================
# 邏輯核心類別 - 超靈敏優化版
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
        
        self.state = 'WAIT_GUARD' 
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        
        # 數據記錄
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.max_v_temp = 0.0
        self.last_hand = "None"
        self.trigger_reason = "" 
        
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心參數：降低 30% 門檻並強化手肘判定 ===
        self.EXTENSION_THRESHOLD = 0.056     # 原 0.08 -> 降低 30% (只要手腕稍微移動即觸發)
        self.RETRACTION_THRESHOLD = 0.18    # 原 0.15 -> 放寬歸位判定，讓下一拳準備更快
        self.ELBOW_LIFT_THRESHOLD = 0.40    # 原 0.25 -> 提高手肘權重 (腋下張開即算有效)
        self.ARM_ANGLE_THRESHOLD = 95       # 原 110 -> 只要手臂微張即視為出拳

        # 用於儀表板顯示
        self.current_extension = 0.0
        self.current_angle = 0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]) # Shoulder
        b = np.array([b.x, b.y]) # Elbow
        c = np.array([c.x, c.y]) # Wrist
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dx = landmark.x - prev_landmark.x
        dy = landmark.y - prev_landmark.y
        dz = landmark.z - prev_landmark.z
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        return (dist_px * scale) / dt

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        top_y = max(0, h - 180)
        cv2.rectangle(overlay, (10, top_y), (350, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.state == 'WAIT_GUARD': status_text, color = "RESET: HANDS BACK", (0, 165, 255)
        elif self.state == 'STIMULUS': status_text, color = "GO !!!", (0, 0, 255)
        elif self.state == 'RESULT': status_text, color = "HIT!", (0, 255, 0)
        else: status_text, color = "READY...", (0, 255, 255)
            
        cv2.putText(image, status_text, (20, h - 140), font, 0.8, color, 2)
        r_time_str = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        vel_str = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        cv2.putText(image, f"Time: {r_time_str}", (20, h - 100), font, 0.9, (255, 255, 255), 2)
        cv2.putText(image, f"Speed: {vel_str}", (20, h - 60), font, 0.8, (255, 255, 255), 2)
        
        if self.state == 'RESULT':
            cv2.putText(image, f"Trigger: {self.trigger_reason}", (20, h - 25), font, 0.6, (200, 200, 200), 1)

        # 右側角度 Debug 條 (紅線在 95 度)
        bar_x, bar_w, bar_h, bar_y = 370, 150, 15, h - 40
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 1)
        angle_ratio = self.ARM_ANGLE_THRESHOLD / 180.0
        cv2.line(image, (int(bar_x + angle_ratio*bar_w), bar_y-5), (int(bar_x + angle_ratio*bar_w), bar_y+bar_h+5), (0,0,255), 2)
        fill_len = int((self.current_angle / 180.0) * bar_w)
        fill_color = (0,255,0) if self.current_angle > self.ARM_ANGLE_THRESHOLD else (0,255,255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_len, bar_y + bar_h), fill_color, -1)
        cv2.putText(image, f"Angle: {int(self.current_angle)}", (bar_x, bar_y - 8), font, 0.4, (255,255,255), 1)

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
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            dist_l, dist_r = abs(l_wr.x - l_sh.x), abs(r_wr.x - r_sh.x)
            angle_l, angle_r = self.calculate_angle(l_sh, l_el, l_wr), self.calculate_angle(r_sh, r_el, r_wr)
            self.current_angle = max(angle_l, angle_r)

            sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist if sh_dist > 0 else 0
            l_v, r_v = 0, 0
            if self.prev_landmarks:
                l_v = self.calculate_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_velocity(r_wr, self.prev_landmarks[16], scale, dt)
            self.prev_landmarks = landmarks

            # 狀態機
            if self.state == 'WAIT_GUARD':
                is_retracted = (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD)
                is_hands_up = (l_wr.y < l_sh.y + 0.3)
                if is_retracted and is_hands_up:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.2, 2.5)
                else:
                    cv2.putText(image, "HANDS BACK", (int(w/2)-120, int(h/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state, self.target, self.start_time, self.max_v_temp = 'STIMULUS', random.choice(['LEFT', 'RIGHT']), current_time, 0.0

            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                self.max_v_temp = max(self.max_v_temp, l_v, r_v)

                if elapsed < 0.8:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)

                if elapsed > 3.0: self.state = 'WAIT_GUARD'

                # 靈敏判定核心 (只要滿足一個即中)
                hit, reason = False, ""
                t_wr_x, t_sh_x = (l_wr.x, l_sh.x) if self.target == 'LEFT' else (r_wr.x, r_sh.x)
                t_dist, t_angle = (dist_l, angle_l) if self.target == 'LEFT' else (dist_r, angle_r)
                t_el_y, t_sh_y = (l_el.y, l_sh.y) if self.target == 'LEFT' else (r_el.y, r_sh.y)

                if t_dist > self.EXTENSION_THRESHOLD: hit, reason = True, "Fast Reach"
                elif t_angle > self.ARM_ANGLE_THRESHOLD: hit, reason = True, "Quick Straight"
                elif t_el_y < (t_sh_y + self.ELBOW_LIFT_THRESHOLD): hit, reason = True, "Elbow Up"
                
                if hit:
                    self.last_reaction_time, self.last_velocity, self.last_hand, self.trigger_reason = elapsed * 1000, self.max_v_temp, self.target, reason
                    self.state, self.wait_until = 'RESULT', current_time + 1.5 

            elif self.state == 'RESULT' and current_time > self.wait_until:
                self.state = 'WAIT_GUARD'

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        try:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except: return frame

def main():
    st.set_page_config(page_title="拳擊反應 v9 (極速靈敏)", layout="wide")
    st.sidebar.title("🥊 參數調優")
    st.sidebar.write("- 門檻調低 30%")
    st.sidebar.write("- 手肘高度判定強化")
    st.sidebar.write("- 角度限制放寬至 95°")
    webrtc_streamer(key="boxing-v9", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
