import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

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
        
        # 數據統計
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.record_max_speed = 0.0
        self.reaction_times_list = []
        
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心門檻調優 ===
        self.EXTENSION_THRESHOLD = 0.032     # 手腕位移 (更敏感)
        self.ARM_ANGLE_THRESHOLD = 80        # 手臂張開角度 (放寬)
        self.ELBOW_H_SENSITIVITY = 0.03      # 手肘相對高度提升量 (關鍵：只需提升一點點即判定)
        self.RETRACTION_THRESHOLD = 0.22     # 歸位判定 (更寬鬆，避免卡在預備)

        self.current_intensity = 0.0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def draw_feedback_bar(self, image, h, w):
        """ 右下角出拳幅度回饋條 """
        bar_w, bar_h = 240, 25
        start_x, start_y = w - 260, h - 50
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (30, 30, 30), -1)
        
        fill_w = int(self.current_intensity * bar_w)
        # 漸層色顯示
        color = (0, 255, 0) if self.current_intensity < 0.6 else (0, 200, 255)
        if self.current_intensity >= 0.9: color = (0, 0, 255)
        
        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), color, -1)
        cv2.putText(image, f"PUNCH POWER: {int(self.current_intensity*100)}%", (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 220), (360, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'PRE_START': ("READY...", (0, 255, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT_PENDING': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0))
        }
        text, color = status_map.get(self.state, ("IDLE", (255,255,255)))
        cv2.putText(image, text, (20, h - 185), font, 0.8, color, 2)

        r_time = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        v_speed = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        avg_r = sum(self.reaction_times_list) / len(self.reaction_times_list) if self.reaction_times_list else 0
        
        cv2.putText(image, f"Time: {r_time}", (20, h - 145), font, 0.7, (255,255,255), 2)
        cv2.putText(image, f"Speed: {v_speed}", (20, h - 115), font, 0.7, (255,255,255), 2)
        cv2.line(image, (20, h - 100), (340, h - 100), (100, 100, 100), 1)
        cv2.putText(image, f"MAX: {self.record_max_speed:.1f} m/s", (20, h - 65), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"AVG: {int(avg_r)} ms", (20, h - 35), font, 0.7, (0, 255, 255), 2)

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        self.draw_dashboard(image, h, w)
        self.draw_feedback_bar(image, h, w)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # 座標點
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            # 計算手肘與肩膀垂直差 (Y 越小越高)
            l_elbow_lift = l_sh.y - l_el.y 
            r_elbow_lift = r_sh.y - r_el.y

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
            self.prev_landmarks, self.prev_time = landmarks, current_time

            # 指定目標之即時數值
            t_dist = dist_l if self.target == 'LEFT' else dist_r
            t_angle = angle_l if self.target == 'LEFT' else angle_r
            t_lift = l_elbow_lift if self.target == 'LEFT' else r_elbow_lift
            
            # 強度計算 (回饋條顯示用)
            s_dist = min(1.0, t_dist / self.EXTENSION_THRESHOLD)
            s_angle = min(1.0, (t_angle - 40) / (self.ARM_ANGLE_THRESHOLD - 40))
            s_lift = min(1.0, (t_lift + 0.1) / (self.ELBOW_H_SENSITIVITY + 0.1))
            self.current_intensity = max(s_dist, s_angle, s_lift)

            # --- 狀態機 ---
            if self.state == 'WAIT_GUARD':
                self.current_intensity = 0
                if (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD):
                    self.state, self.wait_until = 'PRE_START', current_time + random
