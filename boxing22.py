import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image # 新增 PIL 用於繪製中文

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
        self.reaction_history = [] 
        self.speed_history = []    
        
        # 顯示控制
        self.show_results = False
        
        # FPS 監測
        self.prev_time = 0
        self.current_fps = 0.0
        self.low_fps_warning = False

        self.prev_landmarks = None
        
        # 參數設定
        self.SHOULDER_WIDTH_M = 0.45 
        self.MIN_VELOCITY_THRESHOLD = 1.2 
        self.Z_PUNCH_THRESHOLD = 0.2
        self.ARM_ANGLE_THRESHOLD = 100 
        self.RETRACTION_THRESHOLD = 0.25
        
        self.current_intensity = 0.0
        self.max_v_temp = 0.0

        # === 字型設定 ===
        # 嘗試載入中文字型，請確保目錄下有 'font.ttf' (例如微軟正黑體)
        # 如果找不到，將使用預設英文
        self.font_path = "font.ttf" 
        try:
            # 測試載入，大小設為 20
            ImageFont.truetype(self.font_path, 20)
            self.use_chinese = True
        except:
            print("警告：找不到 font.ttf，將使用英文顯示。請放入中文字型檔。")
            self.use_chinese = False

    def put_chinese_text(self, img, text, pos, color, size=30):
        """ 輔助函式：將 PIL 繪製的文字疊加回 OpenCV 圖像 """
        if not self.use_chinese:
            # 如果沒有中文字型，回退使用 OpenCV 英文
            # 這裡做一個簡單的英文映射避免亂碼，如果傳入的是純中文且無字型，會顯示問號
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size/30, color, 2)
            return img

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
        return (dist_3d * scale) / dt

    def get_speed_rating(self, speed):
        # 根據要求的中文化評價
        if speed < 6.7: return "一般"
        elif speed < 11.0: return "優異"
        elif speed < 13.0: return "專業選手"
        else: return "頂尖選手"

    def get_reaction_rating(self, r_time):
        # 根據要求的中文化評價
        if r_time > 250: return "一般"
        elif r_time >= 120: return "優異"
        else: return "頂尖選手"

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
        
        display_v = self.last_punch_speed if self.state == 'RESULT' else (self.current_intensity * 13.0)
        # 中文顯示
        txt = f"即時速度: {display_v:.1f} m/s"
        image = self.put_chinese_text(image, txt, (start_x, start_y - 30), (255, 255, 255), 20)
        return image

    def draw_dashboard(self, image, h, w):
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 320), (450, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 狀態文字中文化
        status_map = {
            'WAIT_GUARD': ("重置：請舉手護頭", (0, 165, 255)),
            'PRE_START': ("預備...", (0, 255, 255)),
            'STIMULUS': ("開始 !!!", (0, 0, 255)),
            'RESULT_PENDING': ("開始 !!!", (0, 0, 255)),
            'RESULT': ("命中!", (0, 255, 0))
        }
        text, color = status_map.get(self.state, ("閒置", (255,255,255)))
        
        # 顯示狀態 (大字體)
        image = self.put_chinese_text(image, text, (20, h - 280), color, 40)

        # 顯示數據
        if self.show_results:
            r_time_val = int(self.last_reaction_time)
            speed_val = self.last_punch_speed
            r_rating = self.get_reaction_rating(r_time_val)
            s_rating = self.get_speed_rating(speed_val)

            # 反應時間 + 評價
            line1 = f"反應時間: {r_time_val} ms [{r_rating}]"
            image = self.put_chinese_text(image, line1, (20, h - 220), (255, 255, 255), 24)

            # 出拳速度 + 評價
            line2 = f"出拳速度: {speed_val:.1f} m/s [{s_rating}]"
            image = self.put_chinese_text(image, line2, (20, h - 180), (255, 255, 255), 24)

            # 分隔線
            cv2.line(image, (20, h - 160), (430, h - 160), (100, 100, 100), 1)

            # 平均數據
            avg_time = np.mean(self.reaction_history) if self.reaction_history else 0
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0
            
            line3 = f"平均反應: {int(avg_time)} ms"
            line4 = f"平均速度: {avg_speed:.1f} m/s"
            
            image = self.put_chinese_text(image, line3, (20, h - 130), (150, 255, 150), 20)
            image = self.put_chinese_text(image, line4, (20, h - 90), (150, 255, 150), 20)

        # 低 FPS 警告 (中文化)
        if self.low_fps_warning:
            warn1 = "警告：偵率過低 (Low FPS)"
            warn2 = f"目前 FPS: {int(self.current_fps)} (建議: 60)，可能影響準確度"
            image = self.put_chinese_text(image, warn1, (20, h - 60), (0, 255, 255), 18)
            image = self.put_chinese_text(image, warn2, (20, h - 35), (0, 255, 255), 18)
            
        return image

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
            
            sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 0

            curr_v = 0.0
            if self.prev_landmarks and dt > 0:
                l_v = self.calculate_3d_velocity(l_wr, self.prev_landmarks[15], scale, dt)
                r_v = self.calculate_3d_velocity(r_wr, self.prev_landmarks[16], scale, dt)
                curr_v = max(l_v, r_v)
            
            display_v = curr_v if curr_v > self.MIN_VELOCITY_THRESHOLD else 0.0
            self.current_intensity = min(1.0, display_v / 13.0) 

            self.prev_landmarks, self.prev_time = landmarks, current_time
            
            # --- 狀態機 ---
            dist_l_2d = abs(l_wr.x - l_sh.x)
            dist_r_2d = abs(r_wr.x - r_sh.x)

            if self.state == 'WAIT_GUARD':
                if (dist_l_2d < self.RETRACTION_THRESHOLD) and (dist_r_2d < self.RETRACTION_THRESHOLD):
                    self.state, self.wait_until = 'PRE_START', current_time + random.uniform(1.5, 3.0)
                else:
                    # 畫面中央提示
                    image = self.put_chinese_text(image, "請舉手!", (int(w/2)-80, h-100), (255, 255, 255), 50)

            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.0
                    self.max_v_temp = 0.0
                    self.show_results = False

            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 0, 255) if self.target == 'LEFT' else (255, 0, 0)
                    # 顯示 左拳! / 右拳!
                    target_text = "左拳!" if self.target == 'LEFT' else "右拳!"
                    # 因為字體大小問題，這裡手動調整位置
                    image = self.put_chinese_text(image, target_text, (int(w/2)-120, int(h/2)-50), color, 100)

            if self.state == 'STIMULUS':
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
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    self.last_punch_speed = max(self.max_v_temp, curr_v)
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
            # 因為用了 PIL，處理稍微變慢，但為了中文顯示是必須的
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except Exception as e: 
            print(e)
            return frame

def main():
    st.set_page_config(page_title="拳擊反應 v19 (中文評測版)", layout="wide")
    st.title("🥊 拳擊反應 - 中文評測版")
    st.sidebar.write("v19 更新說明：")
    st.sidebar.write("1. 介面全中文化 (含手機畫面)")
    st.sidebar.write("2. 速度評價：一般 / 優異 / 專業 / 頂尖")
    st.sidebar.write("3. 增加平均反應與平均速度統計")
    st.sidebar.write("⚠️ 注意：伺服器端必須有 font.ttf 才能顯示中文，否則將顯示預設英文")
    
    webrtc_streamer(
        key="boxing-v19-cn", 
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
