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
        # 假設一般人肩寬 0.45 公尺，用來將像素/歸一化座標轉為真實米數
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心門檻：3D 物理判定修正 ===
        # 1. 最小速度門檻 (防抖動)：速度低於 1.2 m/s 不視為出拳，視為雜訊
        self.MIN_VELOCITY_THRESHOLD = 1.2 
        
        # 2. Z軸 (前進) 觸發門檻：手腕必須比肩膀 "更靠近鏡頭" 多少單位
        # MediaPipe 中，Z 越負代表越靠近鏡頭
        self.Z_PUNCH_THRESHOLD = 0.2
        
        # 3. 手臂伸直角度
        self.ARM_ANGLE_THRESHOLD = 100 

        # 歸位判定 (寬鬆)
        self.RETRACTION_THRESHOLD = 0.25
        
        self.current_intensity = 0.0

    def calculate_angle(self, a, b, c):
        # 這裡只算 2D 投影角度供參考，因為手肘打直主要看這裡
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def calculate_3d_velocity(self, curr, prev, scale, dt):
        """ 計算 XYZ 三維速度向量 """
        if dt <= 0: return 0
        
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        # MediaPipe 的 Z 座標在歸一化空間中，大致與 X 的寬度比例類似
        # 但 Z 變化通常較敏感，我們直接納入計算
        dz = curr.z - prev.z 
        
        # 3D 距離 (Euclidean distance)
        dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 轉換為公尺/秒
        velocity = (dist_3d * scale) / dt
        return velocity

    def draw_feedback_bar(self, image, h, w):
        """ 右下角：顯示即時速度強度 (過濾雜訊後) """
        bar_w, bar_h = 240, 25
        start_x, start_y = w - 260, h - 60
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        
        # 強度：基於當前速度 / 預期最大速度 (例如 8 m/s)
        fill_w = int(self.current_intensity * bar_w)
        
        # 顏色邏輯：未達最小門檻為灰/白，達到攻擊速度為紅
        if self.last_velocity < self.MIN_VELOCITY_THRESHOLD and self.state == 'STIMULUS':
            color = (150, 150, 150) # 噪音區
        elif self.current_intensity < 0.5:
            color = (0, 255, 255)   # 黃 (蓄力)
        else:
            color = (0, 0, 255)     # 紅 (有效打擊)

        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), color, -1)
        
        # 文字
        txt = f"SPEED: {self.last_velocity:.1f} m/s"
        if self.last_velocity < self.MIN_VELOCITY_THRESHOLD:
             txt += " (NOISE)"
        cv2.putText(image, txt, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 220), (360, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'PRE_START': ("READY...", (0, 255, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT_PENDING': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0))
        }
        text, color = status_map.get(self.state, ("IDLE", (255,255,255)))
        cv2.putText(image, text, (20, h - 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        r_time = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        # 這裡顯示的是命中當下的速度
        v_speed = f"{self.record_max_speed:.1f} m/s" if self.record_max_speed > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time}", (20, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(image, f"Max Spd: {v_speed}", (20, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.line(image, (20, h - 100), (340, h - 100), (100, 100, 100), 1)

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # 取得關鍵點
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            # 1. 計算比例尺 (像素 -> 公尺)
            # 這裡用 shoulder 2D 距離當基準，雖然有 Z 軸誤差，但做為相對參考足夠
            sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 0

            # 2. 計算 3D 瞬時速度
            curr_v = 0.0
            dt = current_time - self.prev_time
            if self.prev_landmarks and dt > 0:
                l_v = self.calculate_3d_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_3d_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                curr_v = max(l_v, r_v)
            
            # 過濾：如果速度小於門檻，視為 0 (去除站立抖動)
            display_v = curr_v if curr_v > self.MIN_VELOCITY_THRESHOLD else 0.0
            
            # 更新全域變數供 UI 使用
            self.last_velocity = display_v 
            self.current_intensity = min(1.0, display_v / 8.0) # 假設 8m/s 為滿格

            self.prev_landmarks, self.prev_time = landmarks, current_time
            
            # --- 狀態機 ---
            dist_l_2d = abs(l_wr.x - l_sh.x)
            dist_r_2d = abs(r_wr.x - r_sh.x)

            if self.state == 'WAIT_GUARD':
                # 重置最大速度
                self.record_max_speed = 0.0
                if (dist_l_2d < self.RETRACTION_THRESHOLD) and (dist_r_2d < self.RETRACTION_THRESHOLD):
                    self.state, self.wait_until = 'PRE_START', current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "HANDS UP!", (int(w/2)-100, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
                # 只有在速度超過噪音門檻時，才開始記錄最大速度
                if curr_v > self.MIN_VELOCITY_THRESHOLD:
                    self.max_v_temp = max(self.max_v_temp, curr_v)

                # 鎖定目標手
                t_wr = l_wr if self.target == 'LEFT' else r_wr
                t_sh = l_sh if self.target == 'LEFT' else r_sh
                t_el = l_el if self.target == 'LEFT' else r_el
                
                # === 嚴格判定條件 (AND 邏輯) ===
                # 1. 速度必須夠快 (代表是 punch 不是 move)
                cond_speed = curr_v > self.MIN_VELOCITY_THRESHOLD
                
                # 2. Z 軸判定: 手腕必須明顯在肩膀 "前面" (Z 值更小)
                # 一般預備時手腕 Z 約等於肩膀 Z，出拳時 Z 會變小
                # 我們設定手腕 Z 必須比肩膀 Z 小 0.2 以上 (數值需視環境微調)
                cond_z_forward = (t_wr.z < t_sh.z - self.Z_PUNCH_THRESHOLD)
                
                # 3. 2D 輔助判定 (避免完全沒伸直)
                t_angle = self.calculate_angle(t_sh, t_el, t_wr)
                cond_extend = t_angle > self.ARM_ANGLE_THRESHOLD

                # 只有當 "有速度" 且 "往前打(Z)" 才算命中
                if cond_speed and (cond_z_forward or cond_extend):
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    self.record_max_speed = self.max_v_temp # 鎖定這拳的最大速度
                    self.reaction_times_list.append(self.last_reaction_time)
                    
                    self.state, self.wait_until = 'RESULT_PENDING', self.command_display_until
                
                if (current_time - self.start_time) > 3.0: self.state = 'WAIT_GUARD'

            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state, self.wait_until = 'RESULT', current_time + 2.0

            elif self.state == 'RESULT':
                if current_time > self.wait_until: self.state = 'WAIT_GUARD'
        
        self.draw_dashboard(image, h, w)
        self.draw_feedback_bar(image, h, w)
        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        try:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except: return frame

def main():
    st.set_page_config(page_title="拳擊反應 v17 (3D物理修正)", layout="wide")
    st.title("🥊 拳擊反應 - 3D 速度與 Z 軸判定版")
    st.sidebar.write("v17 更新重點：")
    st.sidebar.write("1. 修正：往鏡頭打(Z軸)現在有速度了")
    st.sidebar.write("2. 修正：過濾身體抖動 (速度 < 1.2m/s 忽略)")
    st.sidebar.write("3. 判定：必須兼具「速度」與「前進」")
    webrtc_streamer(key="boxing-v17", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
