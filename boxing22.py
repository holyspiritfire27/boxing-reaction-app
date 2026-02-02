import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

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
        
        # 第一次測試 1.0秒，之後 0.5秒
        self.is_first_run = True 
        self.guard_hold_start_time = None 
        
        # 數據統計
        self.last_reaction_time = 0.0
        self.last_punch_speed = 0.0
        self.reaction_history = [] 
        self.speed_history = []    
        self.show_results = False
        
        # FPS 監測
        self.prev_time = 0
        self.current_fps = 0.0
        self.low_fps_warning = False

        self.prev_landmarks = None
        
        # 參數設定
        self.SHOULDER_WIDTH_M = 0.45 
        
        # === 核心物理修正參數 ===
        self.MIN_VELOCITY_THRESHOLD = 2.0  # 觸發偵測的門檻
        self.ACC_WINDOW = 0.25             # 僅計算前 0.25 秒的爆發
        self.Z_PUNCH_THRESHOLD = 0.15      # 擊中判定的深度
        self.ARM_ANGLE_THRESHOLD = 120     # 手臂打直角度
        self.RETRACTION_THRESHOLD = 0.30 
        
        # 速度計算變數
        self.acc_start_time = None         # 加速期開始時間
        self.max_v_temp = 0.0              # 當次揮拳最大速度
        self.prev_instant_v = 0.0          # 上一幀的瞬時速度 (判斷加速度用)

        # 字型設定
        self.font_path = "font.ttf" 
        try:
            ImageFont.truetype(self.font_path, 20)
            self.use_chinese = True
        except:
            self.use_chinese = False

    def put_chinese_text(self, img, text, pos, color, size=30, stroke_width=0, stroke_fill=(0,0,0)):
        if not self.use_chinese:
            cv2_color = (color[2], color[1], color[0]) 
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size/30, cv2_color, 2)
            return img
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, size)
        draw.text(pos, text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def calculate_forward_velocity(self, curr, prev, scale, dt):
        """
        修正優化 3: 只計算 Z 軸向前速度 (Forward Speed)
        """
        if dt <= 0: return 0
        
        # MediaPipe Z 軸: 數值越小代表越靠近鏡頭
        # 所以 prev.z - curr.z > 0 代表向前衝
        dz = prev.z - curr.z 
        
        # 過濾掉向後收拳或不動的雜訊 (只取 > 0)
        forward_dist = max(0, dz)
        
        velocity = (forward_dist * scale) / dt
        return velocity

    def get_speed_rating(self, speed):
        """
        修正優化: 根據新的物理標準更新評價
        慢速推手/一般學生: < 7
        校隊等級: 8-10
        選手級: 10-13
        職業級: > 13
        """
        if speed < 5.0: return "慢速/暖身"
        elif speed < 8.0: return "一般水準"
        elif speed < 11.0: return "校隊等級"
        elif speed < 13.0: return "選手級"
        else: return "職業拳手"

    def get_reaction_rating(self, r_time):
        if r_time > 250: return "一般"
        elif r_time >= 120: return "優異"
        else: return "頂尖選手"

    def draw_feedback_bar(self, image, h, w):
        bar_w, bar_h = 240, 25
        start_x, start_y = w - 260, h - 60
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        
        # 顯示邏輯：
        # 如果正在出拳(STIMULUS)，顯示目前抓到的最大爆發速度
        # 如果結束，顯示最後結果
        display_val = self.max_v_temp if self.state == 'STIMULUS' else self.last_punch_speed
        
        # 職業選手約 16m/s 為滿格
        display_ratio = min(1.0, display_val / 16.0)
        fill_w = int(display_ratio * bar_w)
        
        if display_ratio < 0.4: color = (0, 255, 255) # Cyan
        else: color = (255, 0, 0) # Red

        cv2_color = (color[2], color[1], color[0])
        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), cv2_color, -1)
        
        txt = f"速度峰值: {display_val:.1f} m/s"
        image = self.put_chinese_text(image, txt, (start_x, start_y - 30), (255, 255, 255), 20)
        return image

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 320), (450, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        status_text = "閒置"
        status_color = (255, 255, 255)

        if self.state == 'WAIT_GUARD':
            if self.guard_hold_start_time is not None:
                elapsed = time.time() - self.guard_hold_start_time
                target_duration = 1.0 if self.is_first_run else 0.5
                progress = min(100, int((elapsed / target_duration) * 100))
                status_text = f"保持姿勢... {progress}%"
                status_color = (0, 255, 255) 
            else:
                status_text = "請舉手護頭"
                status_color = (0, 165, 255) 
        elif self.state == 'PRE_START':
            status_text = "預備..."
            status_color = (0, 255, 255)
        elif self.state in ['STIMULUS', 'RESULT_PENDING']:
            status_text = "開始 !!!"
            status_color = (255, 50, 50) 
        elif self.state == 'RESULT':
            status_text = "命中!"
            status_color = (0, 255, 0) 

        image = self.put_chinese_text(image, status_text, (20, h - 280), status_color, 40)

        if self.show_results:
            r_time_val = int(self.last_reaction_time)
            speed_val = self.last_punch_speed
            r_rating = self.get_reaction_rating(r_time_val)
            s_rating = self.get_speed_rating(speed_val)

            image = self.put_chinese_text(image, f"反應時間: {r_time_val} ms [{r_rating}]", (20, h - 220), (255, 255, 255), 24)
            image = self.put_chinese_text(image, f"出拳速度: {speed_val:.1f} m/s [{s_rating}]", (20, h - 180), (255, 255, 255), 24)
            cv2.line(image, (20, h - 160), (430, h - 160), (100, 100, 100), 1)

            avg_time = np.mean(self.reaction_history) if self.reaction_history else 0
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0
            
            image = self.put_chinese_text(image, f"平均反應: {int(avg_time)} ms", (20, h - 130), (150, 255, 150), 20)
            image = self.put_chinese_text(image, f"平均速度: {avg_speed:.1f} m/s", (20, h - 90), (150, 255, 150), 20)

        if self.low_fps_warning:
            image = self.put_chinese_text(image, "警告：FPS 過低", (20, h - 60), (0, 255, 255), 18)
            
        return image

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        dt = current_time - self.prev_time
        if dt > 0:
            self.current_fps = 1.0 / dt
            if self.current_fps < 45: self.low_fps_warning = True
            else: self.low_fps_warning = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_wr, r_wr = landmarks[15], landmarks[16]
            
            sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 0

            # === 核心：物理引擎計算 ===
            forward_v = 0.0
            
            if self.prev_landmarks and dt > 0:
                # 修正優化 3: 使用 Z 軸前衝速度
                l_v = self.calculate_forward_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_forward_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                forward_v = max(l_v, r_v)
            
            # --- 狀態機與速度採樣 ---
            dist_l_2d = abs(l_wr.x - l_sh.x)
            dist_r_2d = abs(r_wr.x - r_sh.x)

            if self.state == 'WAIT_GUARD':
                is_in_guard = (dist_l_2d < self.RETRACTION_THRESHOLD) and (dist_r_2d < self.RETRACTION_THRESHOLD)
                
                if is_in_guard:
                    if self.guard_hold_start_time is None:
                        self.guard_hold_start_time = current_time
                    else:
                        required_duration = 1.0 if self.is_first_run else 0.5
                        if (current_time - self.guard_hold_start_time) > required_duration:
                            self.state, self.wait_until = 'PRE_START', current_time + random.uniform(1.5, 3.0)
                            self.guard_hold_start_time = None
                            self.is_first_run = False 
                else:
                    self.guard_hold_start_time = None
                    image = self.put_chinese_text(image, "請舉手!", (int(w/2)-80, h-100), (255, 255, 255), 50, stroke_width=3)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.0
                    
                    # 重置速度採樣變數
                    self.max_v_temp = 0.0 
                    self.acc_start_time = None # 修正優化 2: 重置加速時間窗
                    self.prev_instant_v = 0.0
                    self.show_results = False

            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 255, 255) if self.target == 'LEFT' else (255, 50, 50)
                    target_text = "左拳!" if self.target == 'LEFT' else "右拳!"
                    image = self.put_chinese_text(image, target_text, (int(w/2)-120, int(h/2)-50), color, 100, stroke_width=6)

            if self.state == 'STIMULUS':
                # === 修正優化 1 & 2: 智慧速度採樣 ===
                
                # 1. 只有當速度大於門檻(開始出拳)時，才啟動計時器
                if forward_v > self.MIN_VELOCITY_THRESHOLD:
                    if self.acc_start_time is None:
                        self.acc_start_time = current_time
                
                # 2. 如果計時器已啟動
                if self.acc_start_time is not None:
                    acc_duration = current_time - self.acc_start_time
                    
                    # 修正優化 2: 只有在 0.25 秒的時間窗內才視為有效爆發
                    if acc_duration < self.ACC_WINDOW:
                        
                        # 修正優化 1: 只有在加速期 (當前速度 > 上一幀速度) 才更新最大值
                        if forward_v > self.prev_instant_v:
                            self.max_v_temp = max(self.max_v_temp, forward_v)
                
                # 更新上一幀速度給下一次比較用
                self.prev_instant_v = forward_v

                # --- 判定擊中 ---
                t_wr = l_wr if self.target == 'LEFT' else r_wr
                t_sh = l_sh if self.target == 'LEFT' else r_sh
                t_el = l_el if self.target == 'LEFT' else r_el
                
                # 判定條件: 
                # 1. 曾經有達到一定速度 (max_v_temp > 門檻)
                # 2. 手臂打直 或 拳頭Z軸明顯前伸
                cond_speed = self.max_v_temp > self.MIN_VELOCITY_THRESHOLD
                cond_z_forward = (t_wr.z < t_sh.z - self.Z_PUNCH_THRESHOLD)
                t_angle = self.calculate_angle(t_sh, t_el, t_wr)
                cond_extend = t_angle > self.ARM_ANGLE_THRESHOLD

                if cond_speed and (cond_z_forward or cond_extend):
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    
                    # 避免極端雜訊
                    if self.max_v_temp > 25.0: self.last_punch_speed = forward_v
                    else: self.last_punch_speed = self.max_v_temp

                    self.reaction_history.append(self.last_reaction_time)
                    self.speed_history.append(self.last_punch_speed)
                    
                    self.show_results = True
                    self.state, self.wait_until = 'RESULT_PENDING', self.command_display_until
                
                if (current_time - self.start_time) > 3.0: 
                    self.state = 'WAIT_GUARD'
                    self.show_results = True 

            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state, self.wait_until = 'RESULT', current_time + 2.0

            elif self.state == 'RESULT':
                if current_time > self.wait_until: self.state = 'WAIT_GUARD'
            
            self.prev_landmarks, self.prev_time = landmarks, current_time
        
        else:
            self.prev_time = current_time

        image = self.draw_dashboard(image, h, w)
        image = self.draw_feedback_bar(image, h, w)
        return image

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        try:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except Exception as e: 
            print(e)
            return frame

def main():
    st.set_page_config(page_title="拳擊反應 v23 (物理引擎修正版)", layout="wide")
    st.title("🥊 拳擊反應 - 物理引擎修正版")
    st.sidebar.write("v23 專業物理修正：")
    st.sidebar.write("1. 僅計算「加速期」速度 (區分推/打)")
    st.sidebar.write("2. 限制 0.25秒 爆發時間窗")
    st.sidebar.write("3. 鎖定 Z 軸前衝速度 (抗側移雜訊)")
    st.sidebar.write("4. 更新職業級速度評價標準")
    
    webrtc_streamer(
        key="boxing-v23-physics", 
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
