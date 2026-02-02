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
        
        # 數據統計與歷史紀錄
        self.last_reaction_time = 0.0
        self.last_punch_speed = 0.0
        self.reaction_history = [] # 儲存歷史反應時間
        self.speed_history = []    # 儲存歷史速度
        
        # 顯示控制
        self.show_results = False  # 控制何時顯示數據
        
        # FPS 監測
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.low_fps_warning = False

        self.prev_landmarks = None
        self.prev_time = 0
        # 假設一般人肩寬 0.45 公尺
        self.SHOULDER_WIDTH_M = 0.45 

        # === 核心門檻 ===
        self.MIN_VELOCITY_THRESHOLD = 1.2 
        self.Z_PUNCH_THRESHOLD = 0.2
        self.ARM_ANGLE_THRESHOLD = 100 
        self.RETRACTION_THRESHOLD = 0.25
        
        self.current_intensity = 0.0
        self.max_v_temp = 0.0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def calculate_3d_velocity(self, curr, prev, scale, dt):
        if dt <= 0: return 0
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dz = curr.z - prev.z 
        dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        velocity = (dist_3d * scale) / dt
        return velocity

    def get_speed_rating(self, speed):
        """ 速度評價邏輯 """
        if speed < 6.7: return "Normal (一般)"
        elif speed < 11.0: return "Excellent (優異)"
        elif speed < 13.0: return "Pro (專業選手)"
        else: return "Elite (頂尖選手)"

    def get_reaction_rating(self, r_time):
        """ 反應時間評價邏輯 (ms) """
        if r_time > 250: return "Normal (一般)"
        elif r_time >= 120: return "Excellent (優異)"
        else: return "Elite (頂尖選手)"

    def draw_feedback_bar(self, image, h, w):
        bar_w, bar_h = 240, 25
        start_x, start_y = w - 260, h - 60
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        fill_w = int(self.current_intensity * bar_w)
        
        if self.current_intensity == 0 and self.state == 'STIMULUS':
             color = (150, 150, 150)
        elif self.current_intensity < 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), color, -1)
        # 顯示即時速度 (僅供參考用)
        # 實際判定用的是 max_v_temp
        display_v = self.last_punch_speed if self.state == 'RESULT' else (self.current_intensity * 8.0)
        txt = f"INSTANT v: {display_v:.1f} m/s"
        cv2.putText(image, txt, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_dashboard(self, image, h, w):
        # 半透明背景板 (加大以容納更多資訊)
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 300), (420, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 狀態文字
        status_map = {
            'WAIT_GUARD': ("RESET: HANDS UP", (0, 165, 255)),
            'PRE_START': ("READY...", (0, 255, 255)),
            'STIMULUS': ("GO !!!", (0, 0, 255)),
            'RESULT_PENDING': ("GO !!!", (0, 0, 255)),
            'RESULT': ("HIT!", (0, 255, 0))
        }
        text, color = status_map.get(self.state, ("IDLE", (255,255,255)))
        cv2.putText(image, text, (20, h - 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 顯示數據 (僅在 show_results 為 True 時顯示，即揮拳後)
        if self.show_results:
            # 1. 本次數據
            r_time_val = int(self.last_reaction_time)
            speed_val = self.last_punch_speed
            
            r_rating = self.get_reaction_rating(r_time_val)
            s_rating = self.get_speed_rating(speed_val)

            # 反應時間 + 評價
            cv2.putText(image, f"Time: {r_time_val} ms", (20, h - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(image, f"[{r_rating}]", (20, h - 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # 出拳速度 + 評價
            cv2.putText(image, f"Speed: {speed_val:.1f} m/s", (20, h - 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(image, f"[{s_rating}]", (20, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # 分隔線
            cv2.line(image, (20, h - 125), (400, h - 125), (100, 100, 100), 1)

            # 2. 平均數據
            avg_time = np.mean(self.reaction_history) if self.reaction_history else 0
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0
            
            cv2.putText(image, f"Avg Time: {int(avg_time)} ms", (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,255,150), 1)
            cv2.putText(image, f"Avg Speed: {avg_speed:.1f} m/s", (20, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,255,150), 1)

        # 低 FPS 警告 (隨時顯示)
        if self.low_fps_warning:
            cv2.putText(image, "WARNING: Low FPS! Accuracy may be low.", (20, h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(image, f"FPS: {int(self.current_fps)} (Target: 60)", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        # === 計算 FPS ===
        dt = current_time - self.prev_time
        if dt > 0:
            self.current_fps = 1.0 / dt
            # 簡單濾波，如果持續低於 45FPS 則警告 (保留一點寬容度)
            if self.current_fps < 45: 
                self.low_fps_warning = True
            else:
                self.low_fps_warning = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            # 計算比例尺
            sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 0

            # 計算瞬時速度
            curr_v = 0.0
            if self.prev_landmarks and dt > 0:
                l_v = self.calculate_3d_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_3d_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                curr_v = max(l_v, r_v)
            
            # 過濾身體抖動
            display_v = curr_v if curr_v > self.MIN_VELOCITY_THRESHOLD else 0.0
            self.current_intensity = min(1.0, display_v / 13.0) # 修正為 13m/s 滿格

            self.prev_landmarks, self.prev_time = landmarks, current_time
            
            # --- 狀態機 ---
            dist_l_2d = abs(l_wr.x - l_sh.x)
            dist_r_2d = abs(r_wr.x - r_sh.x)

            if self.state == 'WAIT_GUARD':
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
                    self.show_results = False # 進入 Hit 階段，隱藏舊數據

            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    cv2.putText(image, f"{self.target}!", (int(w/2)-120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

            if self.state == 'STIMULUS':
                # 紀錄最大速度 (Peak Velocity)
                if curr_v > self.MIN_VELOCITY_THRESHOLD:
                    self.max_v_temp = max(self.max_v_temp, curr_v)

                t_wr = l_wr if self.target == 'LEFT' else r_wr
                t_sh = l_sh if self.target == 'LEFT' else r_sh
                t_el = l_el if self.target == 'LEFT' else r_el
                
                cond_speed = curr_v > self.MIN_VELOCITY_THRESHOLD
                cond_z_forward = (t_wr.z < t_sh.z - self.Z_PUNCH_THRESHOLD)
                t_angle = self.calculate_angle(t_sh, t_el, t_wr)
                cond_extend = t_angle > self.ARM_ANGLE_THRESHOLD

                if cond_speed and (cond_z_forward or cond_extend):
                    # 命中！計算數據
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    
                    # 速度採用過程中捕捉到的最大值 (避免擊中瞬間減速造成的低估)
                    # 若 max_v_temp 異常低（例如只有剛好過門檻），則取當前速度
                    self.last_punch_speed = max(self.max_v_temp, curr_v)
                    
                    # 存入歷史紀錄
                    self.reaction_history.append(self.last_reaction_time)
                    self.speed_history.append(self.last_punch_speed)
                    
                    self.show_results = True # 顯示數據
                    self.state, self.wait_until = 'RESULT_PENDING', self.command_display_until
                
                if (current_time - self.start_time) > 3.0: 
                    self.state = 'WAIT_GUARD'
                    self.show_results = True # 超時也顯示(雖然是失敗)

            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state, self.wait_until = 'RESULT', current_time + 2.0

            elif self.state == 'RESULT':
                if current_time > self.wait_until: self.state = 'WAIT_GUARD'
        
        else:
            # 第一幀沒有 landmark 時更新時間，避免 dt 變得極大
            self.prev_time = current_time

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
    st.set_page_config(page_title="拳擊反應 v18 (專業評測版)", layout="wide")
    st.title("🥊 拳擊反應 - 專業評測版")
    st.sidebar.write("v18 更新重點：")
    st.sidebar.write("1. 評分：新增速度與反應分級 (一般/優異/頂尖)")
    st.sidebar.write("2. 統計：新增平均反應時間與平均速度")
    st.sidebar.write("3. 顯示：擊打過程隱藏數據，擊中後顯示")
    st.sidebar.write("4. 效能：自動偵測 FPS，不足時提示")
    
    # 在 media_stream_constraints 加入 frameRate 要求
    # 注意：手機瀏覽器不一定會完全遵守，但這會發出請求
    webrtc_streamer(
        key="boxing-v18", 
        video_processor_factory=VideoProcessor, 
        media_stream_constraints={
            "video": {
                "frameRate": {"ideal": 60, "min": 30},
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            }, 
            "audio": False
        }, 
        async_processing=True
    )

if __name__ == "__main__": main()
