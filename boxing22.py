import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp

# ==========================================
# v7 專業版拳擊分析邏輯
# ==========================================
class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # 狀態
        self.state = "WAIT_GUARD"
        self.target = None
        self.start_time = 0
        self.wait_until = 0

        # 結果數據
        self.last_reaction_time = 0.0
        self.last_velocity = 0.0
        self.last_hand = "None"

        # 追蹤用
        self.prev_landmarks = None
        self.prev_time = time.time()

        # 出拳期間最大速度
        self.max_velocity = 0.0

        # 人體比例
        self.SHOULDER_WIDTH_M = 0.45

        # 判定參數（v7 核心）
        self.EXTENSION_THRESHOLD = 0.13   # 手伸直程度
        self.Z_FORWARD_THRESHOLD = 0.04   # 向前打（關鍵）
        self.RETRACTION_THRESHOLD = 0.15  # 收手

        self.current_extension = 0.0

    # ------------------------
    def calculate_velocity(self, lm, prev_lm, scale, dt):
        if dt <= 0:
            return 0
        dx = lm.x - prev_lm.x
        dy = lm.y - prev_lm.y
        dz = lm.z - prev_lm.z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        return (dist * scale) / dt

    # ------------------------
    def draw_dashboard(self, img, h):
        overlay = img.copy()
        cv2.rectangle(overlay, (10, h-160), (330, h-10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.state == "WAIT_GUARD":
            status = "RESET: HANDS BACK"
            color = (0,165,255)
        elif self.state == "PRE_START":
            status = "READY..."
            color = (0,255,255)
        elif self.state == "STIMULUS":
            status = "GO !!!"
            color = (0,0,255)
        else:
            status = "RESULT"
            color = (0,255,0)

        cv2.putText(img, status, (20, h-120), font, 0.8, color, 2)

        t = f"{int(self.last_reaction_time)} ms" if self.last_reaction_time > 0 else "---"
        v = f"{self.last_velocity:.2f} m/s" if self.last_velocity > 0 else "---"

        cv2.putText(img, f"Time: {t}", (20, h-80), font, 0.9, (255,255,255), 2)
        cv2.putText(img, f"Speed: {v}", (20, h-40), font, 0.9, (255,255,255), 2)

    # ------------------------
    def process(self, image):
        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        self.draw_dashboard(image, h)

        if not results.pose_landmarks:
            return image

        lm = results.pose_landmarks.landmark
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )

        ls, rs = lm[11], lm[12]
        lw, rw = lm[15], lm[16]

        # 伸展量
        dist_l = abs(lw.x - ls.x)
        dist_r = abs(rw.x - rs.x)
        self.current_extension = max(dist_l, dist_r)

        # 比例尺
        shoulder_dist = np.hypot(ls.x - rs.x, ls.y - rs.y)
        scale = self.SHOULDER_WIDTH_M / shoulder_dist if shoulder_dist > 0 else 0

        lv = rv = 0
        if self.prev_landmarks:
            lv = self.calculate_velocity(lw, self.prev_landmarks[15], scale, dt)
            rv = self.calculate_velocity(rw, self.prev_landmarks[16], scale, dt)

        self.prev_landmarks = lm

        # ======================
        # 狀態機 v7
        # ======================
        if self.state == "WAIT_GUARD":
            hands_up = (lw.y < ls.y + 0.2) and (rw.y < rs.y + 0.2)
            retracted = (dist_l < self.RETRACTION_THRESHOLD) and (dist_r < self.RETRACTION_THRESHOLD)

            if hands_up and retracted:
                self.state = "PRE_START"
                self.wait_until = now + random.uniform(1.5, 3.0)

        elif self.state == "PRE_START":
            if now > self.wait_until:
                self.state = "STIMULUS"
                self.target = random.choice(["LEFT", "RIGHT"])
                self.start_time = now
                self.max_velocity = 0.0

        elif self.state == "STIMULUS":
            elapsed = now - self.start_time

            # 指令
            if elapsed < 0.8:
                txt = self.target + "!"
                color = (0,0,255) if self.target=="LEFT" else (255,0,0)
                cv2.putText(image, txt, (w//2-120, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)

            # 更新最大速度
            self.max_velocity = max(self.max_velocity, lv, rv)

            # v7 出拳判定（核心）
            hit = False
            if self.target == "LEFT":
                forward = (ls.z - lw.z) > self.Z_FORWARD_THRESHOLD
                hit = forward and dist_l > self.EXTENSION_THRESHOLD
            else:
                forward = (rs.z - rw.z) > self.Z_FORWARD_THRESHOLD
                hit = forward and dist_r > self.EXTENSION_THRESHOLD

            if hit:
                self.last_reaction_time = elapsed * 1000
                self.last_velocity = self.max_velocity
                self.last_hand = self.target
                self.state = "RESULT"
                self.wait_until = now + 2.0

            if elapsed > 3.0:
                self.state = "WAIT_GUARD"

        elif self.state == "RESULT":
            if now > self.wait_until:
                self.state = "WAIT_GUARD"

        return image


# ==========================================
# Streamlit
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.logic.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="拳擊反應訓練 v7", layout="wide")

    st.sidebar.title("🥊 拳擊反應 v7 專業版")
    st.sidebar.info("""
- **Time (ms)**：指令出現 → 第一次有效出拳  
- **Speed (m/s)**：該拳「最大瞬間速度」

✔ 使用 Z 軸判定真正「往前打拳」  
✔ 濾除假動作與抖動
""")

    st.title("🥊 AI 拳擊反應測試 v7（專業判定）")
    st.markdown("站在鏡頭前約 2 公尺，完整拍到上半身")

    webrtc_streamer(
        key="boxing-v7",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == "__main__":
    main()
