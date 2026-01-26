import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# 邏輯核心類別 - 極致敏銳與循環優化版
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
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
        self.trigger_reason = "" 
        
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心門檻：極致敏感與手肘權重 ===
        self.EXTENSION_THRESHOLD = 0.04      # 極致敏感：手腕稍微離開肩膀即觸發
        self.RETRACTION_THRESHOLD = 0.18     # 歸位判定
        self.ELBOW_LIFT_THRESHOLD = 0.55     # 極大手肘權重：手肘稍微抬起即算有效
        self.ARM_ANGLE_THRESHOLD = 90        # 只要手臂成直角以上即視為攻擊

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dist_px = np.sqrt((landmark.x - prev_landmark.x)**2 + (landmark.y - prev_landmark.y)**2)
        return (dist_px * scale) / dt

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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            dist_l, dist_r = abs(l_wr.x - l_sh.x), abs(r_wr.x - r_sh.x)
            angle_l, angle_r = self.calculate_angle(l_sh, l_el, l_wr), self.calculate_angle(r_sh, r_el, r_wr)

            # 速度計算
            sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist if sh_dist > 0 else 0
            curr_v = 0
            if self.prev_landmarks:
                l_v = self.calculate_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                curr_v = max(l_v, r_v)
            self.prev_landmarks = landmarks

            # ==========================
            # 狀態機循環邏輯
            # ==========================
            
            # 1. 預備狀態 (WAIT_GUARD)
            if self.state == 'WAIT_GUARD':
                is_ready = (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD)
                if is_ready:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                else:
                    cv2.putText(image, "READY: HANDS UP", (int(w/2)-150, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 延遲隨機時間 (PRE_START)
            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.max_v_temp = 0.0

            # 2. 要求出拳 (STIMULUS) - 限制 0.5 秒高亮，但持續監測
            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                self.max_v_temp = max(self.max_v_temp, curr_v)

                # 指令顯示 (僅在 0.5 秒內顯示超大字體，增加壓迫感)
                if elapsed < 0.5:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

                # 判定邏輯
                t_dist = dist_l if self.target == 'LEFT' else dist_r
                t_angle = angle_l if self.target == 'LEFT' else angle_r
                t_el_y, t_sh_y = (l_el.y, l_sh.y) if self.target == 'LEFT' else (r_el.y, r_sh.y)

                hit, reason = False, ""
                if t_dist > self.EXTENSION_THRESHOLD: hit, reason = True, "Fast Reach"
                elif t_angle > self.ARM_ANGLE_THRESHOLD: hit, reason = True, "Quick Straight"
                elif t_el_y < (t_sh_y + self.ELBOW_LIFT_THRESHOLD): hit, reason = True, "Elbow Power"

                if hit:
                    self.last_reaction_time = elapsed * 1000
                    self.last_velocity = self.max_v_temp
                    self.trigger_reason = reason
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0 # 3. 結果呈現 2 秒

                if elapsed > 2.0: # 超過兩秒未出拳自動重置
                    self.state = 'WAIT_GUARD'

            # 3. 結果呈現 (RESULT)
            elif self.state == 'RESULT':
                cv2.putText(image, "HIT!", (int(w/2)-80, int(h/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
                cv2.putText(image, f"{int(self.last_reaction_time)}ms", (int(w/2)-80, int(h/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")

def main():
    st.set_page_config(page_title="拳擊反應 v10 (極速靈敏版)", layout="wide")
    st.title("🥊 AI 拳擊反應訓練 v10.0")
    st.sidebar.info("順序：預備 -> 指令(0.5s) -> 結果(2s) -> 重複")
    webrtc_streamer(key="boxing-v10", video_processor_factory=VideoProcessor, 
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

if __name__ == "__main__": main()
