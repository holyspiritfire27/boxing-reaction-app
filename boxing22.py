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

        # === 核心門檻：手肘高度優先版 ===
        self.ELBOW_LIFT_THRESHOLD = 0.02    # 手肘相對於肩膀的 Y 軸差距 (數值越小越靈敏)
        self.EXTENSION_THRESHOLD = 0.03     # 手腕位移觸發
        self.ARM_ANGLE_THRESHOLD = 80       # 手臂張開角度觸發
        self.RETRACTION_THRESHOLD = 0.25    # 回歸預備動作的寬鬆度

        self.current_intensity = 0.0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def draw_feedback_bar(self, image, h, w):
        """ 右下角出拳幅度回饋條：顯示手肘高度與位移的綜合感應 """
        bar_w, bar_h = 240, 25
        start_x, start_y = w - 260, h - 60
        
        # 外框與背景
        cv2.rectangle(image, (start_x-5, start_y-30), (start_x + bar_w + 5, start_y + bar_h + 5), (0, 0, 0), -1)
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), 2)
        
        # 填充顏色隨進度變化
        fill_w = int(self.current_intensity * bar_w)
        if self.current_intensity < 0.6:
            color = (0, 255, 0) # 綠
        elif self.current_intensity < 0.9:
            color = (0, 255, 255) # 黃
        else:
            color = (0, 0, 255) # 紅
            
        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), color, -1)
        
        # 文字標籤
        percent = int(self.current_intensity * 100)
        cv2.putText(image, f"PUNCH TRIGGER: {percent}%", (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
        v_speed = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        avg_r = sum(self.reaction_times_list) / len(self.reaction_times_list) if self.reaction_times_list else 0
        
        cv2.putText(image, f"React: {r_time}", (20, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(image, f"Speed: {v_speed}", (20, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.line(image, (20, h - 100), (340, h - 100), (100, 100, 100), 1)
        cv2.putText(image, f"MAX SPD: {self.record_max_speed:.1f}", (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"AVG REACT: {int(avg_r)}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

            # 取得關鍵點
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            # 手肘高度計算 (肩膀Y - 手肘Y；手肘越高，此值越大)
            l_elbow_h = l_sh.y - l_el.y
            r_elbow_h = r_sh.y - r_el.y

            dist_l, dist_r = abs(l_wr.x - l_sh.x), abs(r_wr.x - r_sh.x)
            angle_l, angle_r = self.calculate_angle(l_sh, l_el, l_wr), self.calculate_angle(r_sh, r_el, r_wr)

            # 當前速度
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

            # 鎖定當前目標手
            t_dist = dist_l if self.target == 'LEFT' else dist_r
            t_angle = angle_l if self.target == 'LEFT' else angle_r
            t_elbow_h = l_elbow_h if self.target == 'LEFT' else r_elbow_h
            
            # 回饋條強度 (混合手肘高度與位移)
            # 將手肘高度差對應到 0~1 (假設高度差達到 0.05 為滿格)
            s_height = min(1.0, (t_elbow_h + 0.1) / 0.15) 
            s_dist = min(1.0, t_dist / self.EXTENSION_THRESHOLD)
            self.current_intensity = max(s_height, s_dist)

            # --- 狀態切換 ---
            if self.state == 'WAIT_GUARD':
                self.current_intensity = 0
                if (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD):
                    self.state, self.wait_until = 'PRE_START', current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "BACK TO GUARD", (int(w/2)-120, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
                self.max_v_temp = max(self.max_v_temp, curr_v)
                
                # 出拳判定條件 (三選一：高度、角度、位移)
                is_lift = t_elbow_h > (l_sh.y - l_el.y - self.ELBOW_LIFT_THRESHOLD) # 高度提升
                is_straight = t_angle > self.ARM_ANGLE_THRESHOLD # 角度打直
                is_reach = t_dist > self.EXTENSION_THRESHOLD # 距離遠離
                
                if is_lift or is_straight or is_reach:
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    self.last_velocity = self.max_v_temp
                    self.reaction_times_list.append(self.last_reaction_time)
                    self.record_max_speed = max(self.record_max_speed, self.last_velocity)
                    self.state, self.wait_until = 'RESULT_PENDING', self.command_display_until
                
                if (current_time - self.start_time) > 3.0: self.state = 'WAIT_GUARD'

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
    st.set_page_config(page_title="拳擊反應 v16", layout="wide")
    st.title("🥊 拳擊反應 - 敏感度優化 & 實時幅度回饋")
    webrtc_streamer(key="boxing-v16", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
