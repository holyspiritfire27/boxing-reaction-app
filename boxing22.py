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
# 邏輯核心類別
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
        # 記錄觸發原因 (Debug用)
        self.trigger_reason = "" 
        
        self.prev_landmarks = None
        self.prev_time = 0
        self.SHOULDER_WIDTH_M = 0.45 

        # === 判定參數 (極致敏銳) ===
        self.EXTENSION_THRESHOLD = 0.08     # X軸只要動一點點就算
        self.RETRACTION_THRESHOLD = 0.15    # 歸位判定
        self.ELBOW_LIFT_THRESHOLD = 0.25    # 肘部高度容許值 (越小越嚴格)
        self.ARM_ANGLE_THRESHOLD = 110      # 手臂角度大於此值視為伸直

        # 用於進度條顯示
        self.current_extension = 0.0
        self.current_angle = 0
        self.current_elbow_h = 0.0

    # 計算三點角度 (肩-肘-腕)
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]) # Shoulder
        b = np.array([b.x, b.y]) # Elbow
        c = np.array([c.x, c.y]) # Wrist
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def calculate_velocity(self, landmark, prev_landmark, scale, dt):
        if dt <= 0: return 0
        dx = landmark.x - prev_landmark.x
        dy = landmark.y - prev_landmark.y
        dz = landmark.z - prev_landmark.z
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        return (dist_px * scale) / dt

    def draw_dashboard(self, image, h, w):
        """ 繪製詳細儀表板 """
        overlay = image.copy()
        top_y = max(0, h - 180)
        cv2.rectangle(overlay, (10, top_y), (350, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        
        # 狀態
        if self.state == 'WAIT_GUARD': 
            status_text = "RESET: HANDS BACK" 
            color = (0, 165, 255)
        elif self.state == 'STIMULUS': 
            status_text = "GO !!!"
            color = (0, 0, 255)
        elif self.state == 'RESULT':
            status_text = "HIT!"
            color = (0, 255, 0)
        else:
            status_text = "READY..."
            color = (0, 255, 255)
            
        cv2.putText(image, status_text, (20, h - 140), font, 0.8, color, 2)

        # 數據
        if self.last_reaction_time > 0:
            r_time_str = f"{int(self.last_reaction_time)} ms"
        else:
            r_time_str = "---"
        
        vel_str = f"{self.last_velocity:.1f} m/s" if self.last_velocity > 0 else "---"
        
        cv2.putText(image, f"Time: {r_time_str}", (20, h - 100), font, 0.9, white, 2)
        cv2.putText(image, f"Speed: {vel_str}", (20, h - 60), font, 0.8, white, 2)
        
        # 顯示觸發原因 (幫助了解是哪個條件抓到的)
        if self.state == 'RESULT':
            cv2.putText(image, f"By: {self.trigger_reason}", (20, h - 25), font, 0.6, (200, 200, 200), 1)

        # === 右側 Debug 條 (顯示手臂角度) ===
        bar_x = 370
        bar_w = 150
        bar_h = 15
        bar_y = h - 40
        
        # 1. 角度條 (Angle Check)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 1)
        # 閾值線 (110度)
        angle_ratio = self.ARM_ANGLE_THRESHOLD / 180.0
        cv2.line(image, (int(bar_x + angle_ratio*bar_w), bar_y-5), (int(bar_x + angle_ratio*bar_w), bar_y+bar_h+5), (0,0,255), 2)
        # 填充
        curr_ratio = self.current_angle / 180.0
        fill_len = int(curr_ratio * bar_w)
        fill_color = (0,255,0) if self.current_angle > self.ARM_ANGLE_THRESHOLD else (0,255,255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_len, bar_y + bar_h), fill_color, -1)
        cv2.putText(image, f"Angle: {int(self.current_angle)}", (bar_x, bar_y - 8), font, 0.4, white, 1)

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
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 關鍵點
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # 計算距離 (X軸伸展)
            dist_l = abs(left_wrist.x - left_shoulder.x)
            dist_r = abs(right_wrist.x - right_shoulder.x)
            self.current_extension = max(dist_l, dist_r)
            
            # 計算角度 (手臂直不直)
            angle_l = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_r = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            self.current_angle = max(angle_l, angle_r)

            # 計算速度
            shoulder_dist = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                    (left_shoulder.y - right_shoulder.y)**2)
            scale_factor = self.SHOULDER_WIDTH_M / shoulder_dist if shoulder_dist > 0 else 0

            left_v = 0
            right_v = 0
            if self.prev_landmarks:
                left_v = self.calculate_velocity(left_wrist, self.prev_landmarks[15], scale_factor, dt)
                right_v = self.calculate_velocity(right_wrist, self.prev_landmarks[16], scale_factor, dt)
            
            self.prev_landmarks = landmarks

            # ==========================
            # 狀態機
            # ==========================
            if self.state == 'WAIT_GUARD':
                # 歸位判定: 手必須收在身體附近
                # 手腕 X 距離肩膀 X 小於 0.15 且 手臂是彎曲的 (< 100度)
                is_retracted = (dist_l < self.RETRACTION_THRESHOLD) and \
                               (dist_r < self.RETRACTION_THRESHOLD)
                
                # 簡單的舉手判定 (不要太嚴格)
                is_hands_up = (left_wrist.y < left_shoulder.y + 0.25)
                
                if is_retracted and is_hands_up:
                    self.state = 'PRE_START'
                    self.wait_until = current_time + random.uniform(1.5, 3.0)
                    self.max_v_temp = 0.0
                else:
                    if int(current_time * 2) % 2 == 0:
                        cv2.putText(image, "NEXT ROUND", (int(w/2)-150, int(h/2)-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(image, "HANDS BACK", (int(w/2)-150, int(h/2)+50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state = 'STIMULUS'
                    self.target = random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.max_v_temp = 0.0

            elif self.state == 'STIMULUS':
                elapsed = current_time - self.start_time
                
                # 記錄最大速度
                current_max_v = max(left_v, right_v)
                if current_max_v > self.max_v_temp:
                    self.max_v_temp = current_max_v

                # 顯示指令
                if elapsed < 0.8:
                    text = self.target + "!"
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    font_scale = 4
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 8)

                if elapsed > 3.0:
                    self.state = 'WAIT_GUARD'

                # === 判定核心 (OR 邏輯：三選一，中一個就算) ===
                hit = False
                reason = ""
                
                # 選定要檢查的手 (左或右)
                if self.target == 'LEFT':
                    wrist_x = left_wrist.x
                    shoulder_x = left_shoulder.x
                    dist = dist_l
                    angle = angle_l
                    elbow_y = left_elbow.y
                    shoulder_y = left_shoulder.y
                else:
                    wrist_x = right_wrist.x
                    shoulder_x = right_shoulder.x
                    dist = dist_r
                    angle = angle_r
                    elbow_y = right_elbow.y
                    shoulder_y = right_shoulder.y

                # 條件 1: 伸展 (原本的邏輯，但門檻降到 0.08)
                cond_reach = dist > self.EXTENSION_THRESHOLD
                
                # 條件 2: 手肘抬起 (您的建議)
                # 當手肘 Y 座標接近肩膀 Y 座標 (數值變小 = 高度變高)
                # 門檻：手肘比「肩膀+0.25」的位置還高，代表離開肋骨了
                cond_elbow = elbow_y < (shoulder_y + self.ELBOW_LIFT_THRESHOLD)
                
                # 條件 3: 手臂打直 (直拳判定神器)
                # 只要手臂角度超過 110 度
                cond_straight = angle > self.ARM_ANGLE_THRESHOLD

                # 綜合判定
                if cond_reach:
                    hit = True
                    reason = "Reach"
                elif cond_elbow:
                    hit = True
                    reason = "Elbow Lift"
                elif cond_straight:
                    hit = True
                    reason = "Straight"
                
                if hit:
                    self.last_reaction_time = elapsed * 1000
                    self.last_velocity = self.max_v_temp
                    self.last_hand = self.target
                    self.trigger_reason = reason # 顯示是靠什麼判定成功的
                    
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0 

            elif self.state == 'RESULT':
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'

        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

def main():
    st.set_page_config(page_title="拳擊反應 v8 (超高敏度)", layout="wide")
    st.sidebar.title("🥊 拳擊反應 v8.0")
    st.sidebar.info(
        """
        **觸發條件 (滿足其一即可):**
        1. **Elbow Lift**: 手肘抬起 (勾拳/側拳)。
        2. **Straight**: 手臂伸直 (直拳/刺拳)。
        3. **Reach**: 手腕遠離肩膀。
        
        **除錯:**
        - 右下角有 **Angle 條**。
        - 只要綠色條超過紅線，就代表手臂夠直。
        """
    )
    st.title("🥊 AI 拳擊反應 (超靈敏版)")
    webrtc_streamer(
        key="boxing-reaction-v8",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
