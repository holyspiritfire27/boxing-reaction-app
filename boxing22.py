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
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        self.state = 'WAIT_GUARD'
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        self.command_display_until = 0

        self.last_reaction_time = 0
        self.last_velocity = 0
        self.record_max_speed = 0
        self.reaction_times_list = []

        self.prev_landmarks = None
        self.prev_time = time.time()
        self.max_v_temp = 0

        self.SHOULDER_WIDTH_M = 0.45

        # ✅ 真正有效的拳擊判定參數
        self.EXTENSION_THRESHOLD = 0.12
        self.Z_FORWARD_THRESHOLD = 0.05
        self.RETRACTION_THRESHOLD = 0.18
        self.MIN_VELOCITY_THRESHOLD = 1.2  # 過濾抖動

    def calculate_velocity(self, lm, prev_lm, scale, dt):
        if dt <= 0: return 0
        dx = lm.x - prev_lm.x
        dy = lm.y - prev_lm.y
        dz = lm.z - prev_lm.z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        return (dist * scale) / dt

    def process(self, image):
        h, w, _ = image.shape
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return image

        lm = results.pose_landmarks.landmark
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        l_sh, r_sh = lm[11], lm[12]
        l_wr, r_wr = lm[15], lm[16]

        dist_l = abs(l_wr.x - l_sh.x)
        dist_r = abs(r_wr.x - r_sh.x)

        shoulder_dist = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        scale = self.SHOULDER_WIDTH_M / shoulder_dist if shoulder_dist > 0 else 0

        lv = rv = 0
        if self.prev_landmarks:
            lv = self.calculate_velocity(l_wr, self.prev_landmarks[15], scale, dt)
            rv = self.calculate_velocity(r_wr, self.prev_landmarks[16], scale, dt)

        self.prev_landmarks = lm

        # ===== 狀態機 =====
        if self.state == 'WAIT_GUARD':
            if dist_l < self.RETRACTION_THRESHOLD and dist_r < self.RETRACTION_THRESHOLD:
                self.state = 'PRE_START'
                self.wait_until = current_time + random.uniform(1.5, 3)

        elif self.state == 'PRE_START':
            if current_time > self.wait_until:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.start_time = current_time
                self.command_display_until = current_time + 1.0
                self.max_v_temp = 0

        if self.state in ['STIMULUS', 'RESULT_PENDING']:
            if current_time <= self.command_display_until:
                color = (0,0,255) if self.target=='LEFT' else (255,0,0)
                cv2.putText(image, f"{self.target}!", (w//2-120, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, color, 8)

        if self.state == 'STIMULUS':
            self.max_v_temp = max(self.max_v_temp, lv, rv)

            if self.target == 'LEFT':
                forward = (l_sh.z - l_wr.z) > self.Z_FORWARD_THRESHOLD
                reach = dist_l > self.EXTENSION_THRESHOLD
                speed_ok = lv > self.MIN_VELOCITY_THRESHOLD
            else:
                forward = (r_sh.z - r_wr.z) > self.Z_FORWARD_THRESHOLD
                reach = dist_r > self.EXTENSION_THRESHOLD
                speed_ok = rv > self.MIN_VELOCITY_THRESHOLD

            if forward and reach and speed_ok:
                self.last_reaction_time = (current_time - self.start_time) * 1000
                self.last_velocity = self.max_v_temp
                self.reaction_times_list.append(self.last_reaction_time)
                self.record_max_speed = max(self.record_max_speed, self.last_velocity)
                self.state = 'RESULT_PENDING'
                self.wait_until = self.command_display_until

            if current_time - self.start_time > 3:
                self.state = 'WAIT_GUARD'

        elif self.state == 'RESULT_PENDING':
            if current_time > self.wait_until:
                self.state = 'RESULT'
                self.wait_until = current_time + 2

        elif self.state == 'RESULT':
            if current_time > self.wait_until:
                self.state = 'WAIT_GUARD'

        cv2.putText(image, f"Time: {int(self.last_reaction_time) if self.last_reaction_time else '--'} ms",
                    (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(image, f"Speed: {self.last_velocity:.2f} m/s" if self.last_velocity else "Speed: --",
                    (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return image


class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")


def main():
    st.set_page_config(page_title="拳擊反應 v17", layout="wide")
    st.title("🥊 拳擊反應測試 v17（專業穩定版）")
    webrtc_streamer(key="boxing-v17", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)

if __name__ == "__main__":
    main()
