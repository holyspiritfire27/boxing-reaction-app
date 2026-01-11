import cv2
import mediapipe as mp
# [新增] 強制導入 pose 模組，確保它被載入
import mediapipe.solutions.pose as mp_pose_impl 
import mediapipe.solutions.drawing_utils as mp_drawing_impl

import time
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ... (略)

class BoxingAnalystLogic:
    def __init__(self):
        # [修改] 使用剛剛強制導入的變數
        self.mp_pose = mp_pose_impl
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0 
        )
        # [修改] 使用強制導入的繪圖工具
        self.mp_drawing = mp_drawing_impl
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0 
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 2. 參數
        self.REAL_ARM_LENGTH_M = 0.85 
        self.pixel_to_meter_scale = 0.0 
        self.ATTACK_START_VELOCITY = 1.0 
        self.ALPHA = 0.6 

        # 3. 狀態機
        self.state = "CALIBRATION"      
        self.active_hand = None        
        self.target_hand = None        
        self.is_correct = False        
        self.target_box = None
        
        # 數據
        self.accum_elbow_r = 0.0
        self.accum_elbow_l = 0.0
        self.prev_elbow_pos_r = None 
        self.prev_elbow_pos_l = None 
        
        self.t_signal = 0        
        self.t_move_start = 0    
        self.t_hit = 0           
        
        self.val_reaction_time = 0.0
        self.val_movement_time = 0.0
        self.val_peak_speed = 0.0
        
        self.wait_start_time = 0
        self.random_delay = 0
        
        # 速度運算變數
        self.prev_wrist_pos_r = None 
        self.prev_wrist_pos_l = None 
        self.smooth_vel_r = 0.0
        self.smooth_vel_l = 0.0
        
        # 計時器
        self.last_frame_time = time.time()
        self.calibration_timer = 0
        
        # 字型 (網頁版改用預設)
        self.font_path = "arial.ttf" 

    def get_3d_distance_px(self, p1, p2, w, h):
        x1, y1, z1 = p1.x * w, p1.y * h, p1.z * w
        x2, y2, z2 = p2.x * w, p2.y * h, p2.z * w
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def calculate_euclidean_dist(self, curr_pos, prev_pos):
        if prev_pos is None or curr_pos is None: return 0.0
        return math.sqrt((curr_pos[0]-prev_pos[0])**2 + 
                         (curr_pos[1]-prev_pos[1])**2 + 
                         (curr_pos[2]-prev_pos[2])**2)

    def get_smoothed_velocity(self, curr_wrist, prev_wrist, prev_smooth_vel, w, h, dt):
        curr_pos = (curr_wrist.x * w, curr_wrist.y * h, curr_wrist.z * w)
        if prev_wrist is None or dt <= 0:
            return 0.0, curr_pos
        delta_dist_px = self.calculate_euclidean_dist(curr_pos, prev_wrist)
        scale = self.pixel_to_meter_scale if self.pixel_to_meter_scale > 0 else 0
        delta_dist_m = delta_dist_px * scale
        raw_velocity = delta_dist_m / dt
        smooth_velocity = (self.ALPHA * raw_velocity) + ((1 - self.ALPHA) * prev_smooth_vel)
        return smooth_velocity, curr_pos

    def check_guard_pose(self, landmarks, w, h):
        nose = landmarks[0]
        # Index Swap 修正
        rw, lw = landmarks[15], landmarks[16]
        dist_r = self.get_3d_distance_px(rw, nose, w, h)
        dist_l = self.get_3d_distance_px(lw, nose, w, h)
        threshold_px = h * 0.4
        if self.pixel_to_meter_scale > 0:
            threshold_px = (self.REAL_ARM_LENGTH_M * 0.6) / self.pixel_to_meter_scale
        return (dist_r < threshold_px) and (dist_l < threshold_px)

    def put_text(self, img, text, pos, color=(0, 255, 0), size=30):
        # 簡化版文字繪製
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size) 
        except:
            font = ImageFont.load_default()
        
        draw.text(pos, text, font=font, fill=color)
        return np.array(img_pil)

    def process_frame(self, frame):
        # 翻轉 (鏡像)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        curr_time = time.time()
        dt = curr_time - self.last_frame_time
        self.last_frame_time = curr_time
        if dt < 0.0001: dt = 0.033

        results = self.pose.process(frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[0]
            # Index Swap
            mp_rw = lm[15] 
            mp_lw = lm[16]
            mp_rs = lm[11] 
            mp_ls = lm[12] 
            mp_re = lm[13] 
            mp_le = lm[14] 

            # 目標框
            box_s = int(w * 0.15) 
            nx, ny = int(nose.x * w), int(nose.y * h)
            self.target_box = (nx - box_s, ny - box_s, nx + box_s, ny + box_s)

            # ----------- 狀態機 -----------
            if self.state == "CALIBRATION":
                dist_3d = self.get_3d_distance_px(mp_rs, mp_rw, w, h)
                is_extended = (dist_3d > w * 0.35)
                
                if is_extended: self.calibration_timer += dt
                else: self.calibration_timer = 0
                
                if self.calibration_timer > 1.0:
                    self.pixel_to_meter_scale = self.REAL_ARM_LENGTH_M / dist_3d
                    self.state = "SETUP"
                
                frame = self.put_text(frame, "【系統校正】", (50, 50), (0, 255, 255), 40)
                frame = self.put_text(frame, "右手平舉 1 秒", (50, 100), (255, 255, 255), 30)
                prog = int((self.calibration_timer / 1.0) * 300)
                cv2.rectangle(frame, (50, 150), (50 + prog, 180), (0, 255, 0), -1)
                cv2.rectangle(frame, (50, 150), (350, 180), (255, 255, 255), 2)

            elif self.state == "SETUP" or self.state == "RESULT":
                if self.state == "RESULT" and (time.time() - self.t_hit > 2.0): 
                    self.state = "SETUP"

                if self.state == "SETUP" and self.check_guard_pose(lm, w, h):
                    self.state = "WAIT"
                    self.wait_start_time = time.time()
                    self.random_delay = random.uniform(2.0, 4.0)
                    self.val_peak_speed = 0.0
                    self.accum_elbow_r = 0.0
                    self.accum_elbow_l = 0.0
                    self.prev_elbow_pos_r = None
                    self.prev_elbow_pos_l = None
                    self.prev_wrist_pos_r = None
                    self.prev_wrist_pos_l = None
                    self.smooth_vel_r = 0.0
                    self.smooth_vel_l = 0.0

            elif self.state == "WAIT":
                # 背景運算
                _, self.prev_wrist_pos_r = self.get_smoothed_velocity(mp_rw, self.prev_wrist_pos_r, 0, w, h, dt)
                _, self.prev_wrist_pos_l = self.get_smoothed_velocity(mp_lw, self.prev_wrist_pos_l, 0, w, h, dt)
                
                if time.time() - self.wait_start_time > self.random_delay:
                    self.state = "STIMULUS"
                    self.t_signal = time.time() 
                    self.target_hand = random.choice(["右手", "左手"])

            elif self.state == "STIMULUS":
                self.smooth_vel_r, self.prev_wrist_pos_r = self.get_smoothed_velocity(mp_rw, self.prev_wrist_pos_r, self.smooth_vel_r, w, h, dt)
                self.smooth_vel_l, self.prev_wrist_pos_l = self.get_smoothed_velocity(mp_lw, self.prev_wrist_pos_l, self.smooth_vel_l, w, h, dt)
                
                if self.smooth_vel_r > self.ATTACK_START_VELOCITY or self.smooth_vel_l > self.ATTACK_START_VELOCITY:
                    self.t_move_start = time.time()
                    self.state = "PUNCHING"
                    self.prev_elbow_pos_r = (mp_re.x*w, mp_re.y*h, mp_re.z*w)
                    self.prev_elbow_pos_l = (mp_le.x*w, mp_le.y*h, mp_le.z*w)

            elif self.state == "PUNCHING":
                self.smooth_vel_r, self.prev_wrist_pos_r = self.get_smoothed_velocity(mp_rw, self.prev_wrist_pos_r, self.smooth_vel_r, w, h, dt)
                self.smooth_vel_l, self.prev_wrist_pos_l = self.get_smoothed_velocity(mp_lw, self.prev_wrist_pos_l, self.smooth_vel_l, w, h, dt)
                self.val_peak_speed = max(self.val_peak_speed, max(self.smooth_vel_r, self.smooth_vel_l))
                
                curr_re = (mp_re.x*w, mp_re.y*h, mp_re.z*w)
                curr_le = (mp_le.x*w, mp_le.y*h, mp_le.z*w)
                self.accum_elbow_r += self.calculate_euclidean_dist(curr_re, self.prev_elbow_pos_r)
                self.accum_elbow_l += self.calculate_euclidean_dist(curr_le, self.prev_elbow_pos_l)
                self.prev_elbow_pos_r = curr_re
                self.prev_elbow_pos_l = curr_le
                
                hit_r = (self.target_box[0] < mp_rw.x*w < self.target_box[2]) and (self.target_box[1] < mp_rw.y*h < self.target_box[3])
                hit_l = (self.target_box[0] < mp_lw.x*w < self.target_box[2]) and (self.target_box[1] < mp_lw.y*h < self.target_box[3])
                
                if (time.time() - self.t_move_start) > 1.5:
                    self.state = "RESULT"

                if hit_r or hit_l:
                    self.t_hit = time.time()
                    self.val_reaction_time = self.t_move_start - self.t_signal
                    self.val_movement_time = self.t_hit - self.t_move_start
                    
                    if self.accum_elbow_r > self.accum_elbow_l:
                        self.active_hand = "右手"
                    else:
                        self.active_hand = "左手"
                    
                    self.is_correct = (self.active_hand == self.target_hand)
                    self.state = "RESULT"

            # ----------- 繪圖 -----------
            if self.state in ["STIMULUS", "PUNCHING"]:
                color = (0, 0, 255)
                if self.state == "PUNCHING" and self.t_hit > 0: color = (0, 255, 0)
                cv2.rectangle(frame, (self.target_box[0], self.target_box[1]), 
                              (self.target_box[2], self.target_box[3]), color, 3)
                
                if self.state == "STIMULUS" and (time.time() - self.t_signal < 0.5):
                     text = f"出拳: {self.target_hand}!"
                     frame = self.put_text(frame, text, (w//2-100, h//2+120), (255, 255, 255), 50)
                else:
                     cv2.putText(frame, "TARGET", (self.target_box[0], self.target_box[1]-5), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # UI 文字
        if self.state == "SETUP":
            frame = self.put_text(frame, "請護頭 (Guard)", (50, 100), (0, 255, 255), 40)
        elif self.state == "WAIT":
            frame = self.put_text(frame, "等待訊號...", (w//2-100, h//2), (0, 165, 255), 60)
        
        elif self.state == "RESULT" and self.t_hit > 0:
            res_str = "成功 O" if self.is_correct else "失敗 X"
            res_col = (0, 255, 0) if self.is_correct else (255, 0, 0)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (30, 200), (480, 550), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            start_y = 240
            gap = 50
            frame = self.put_text(frame, f"判定: {res_str}", (50, start_y), res_col, 40)
            frame = self.put_text(frame, f"出拳: {self.active_hand}", (50, start_y + gap), (0, 255, 255), 30)
            frame = self.put_text(frame, f"反應: {self.val_reaction_time:.3f} s", (50, start_y + gap*2), (255, 100, 100), 30)
            frame = self.put_text(frame, f"動作: {self.val_movement_time:.3f} s", (50, start_y + gap*3), (100, 255, 100), 30)
            frame = self.put_text(frame, f"速度: {self.val_peak_speed:.2f} m/s", (50, start_y + gap*4), (0, 200, 255), 30)

        return frame

# ==========================================
# Streamlit WebRTC 橋接器
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.logic.process_frame(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ==========================================
# 網頁主程式
# ==========================================
st.set_page_config(page_title="拳擊反應訓練", layout="wide")
st.title("🥊 拳擊反應訓練 (手機網頁版)")

st.write("請允許使用攝影機，並將手機橫放或固定好。")

webrtc_streamer(
    key="boxing",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

