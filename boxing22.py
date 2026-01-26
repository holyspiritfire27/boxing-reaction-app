import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# 邏輯核心類別 - 指令強制停留與統計版
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
        self.command_display_until = 0  # 👈 新增：指令最少顯示到的時間點
        
        # 數據記錄
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.max_v_temp = 0.0
        
        # 統計紀錄
        self.record_max_speed = 0.0
        self.reaction_times_list = []
        
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
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 210), (340, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        
        # 狀態顯示 (RESULT_PENDING 時畫面仍顯示 GO，維持視覺一致性)
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT_PENDING': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0)),
            'PRE_START': ("READY...", yellow)
        }
        status_text, color = status_map.get(self.state, ("IDLE", white))
        cv2.putText(image, status_text, (20, h - 175), font, 0.7, color, 2)

        r_time = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        v_speed = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        avg_r = sum(self.reaction_times_list) / len(self.reaction_times_list) if self.reaction_times_list else 0
        
        cv2.putText(image, f"Last Time: {r_time}", (20, h - 140), font, 0.7, white, 2)
        cv2.putText(image, f"Last Speed: {v_speed}", (20, h - 110), font, 0.7, white, 2)
        cv2.line(image, (20, h - 95), (320, h - 95), (100, 100, 100), 1)
        cv2.putText(image, f"Max Speed: {self.record_max_speed:.1f} m/s", (20, h - 65), font, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Avg React: {int(avg_r)} ms", (20, h - 35), font, 0.7, yellow, 2)

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
                    self.command_display_until = current_time + 1.0 # 👈 強制指令顯示 1 秒
                    self.max_v_temp = 0.0

            # 綜合處理指令顯示 (不論是 STIMULUS 還是 RESULT_PENDING，只要時間未到就顯示)
            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

            if self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                self.max_v_temp = max(self.max_v_temp, curr_v)

                # 判定邏輯
                t_dist = dist_l if self.target == 'LEFT' else dist_r
                t_angle = angle_l if self.target == 'LEFT' else angle_r
                t_el_y, t_sh_y = (l_el.y, l_sh.y) if self.target == 'LEFT' else (r_el.y, r_sh.y)

                hit = False
                if t_dist > self.EXTENSION_THRESHOLD: hit = True
                elif t_angle > self.ARM_ANGLE_THRESHOLD: hit = True
                elif t_el_y < (t_sh_y + self.ELBOW_LIFT_THRESHOLD): hit = True

                if hit:
                    self.last_reaction_time = elapsed * 1000
                    self.last_velocity = self.max_v_temp
                    self.reaction_times_list.append(self.last_reaction_time)
                    if self.last_velocity > self.record_max_speed:
                        self.record_max_speed = self.last_velocity
                    
                    # 👈 關鍵：切換到等待顯示結束的狀態
                    self.state = 'RESULT_PENDING'
                    self.wait_until = self.command_display_until

                if elapsed > 3.0: # 超時重置
                    self.state = 'WAIT_GUARD'

            elif self.state == 'RESULT_PENDING':
                # 指令還在畫面上顯示，等待 1 秒時間到
                if current_time > self.wait_until:
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0 # 切換到結果呈現 2 秒

            elif self.state == 'RESULT':
                # 這裡畫面乾淨，純顯示儀表板數據
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

# ... (其餘 VideoProcessor 與 main 部分保持不變) ...
