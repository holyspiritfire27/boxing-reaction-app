import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import mediapipe as mp
import time
import random
import math
from PIL import ImageFont, ImageDraw, Image

# ================= 配置與常數 =================
st.set_page_config(page_title="拳擊反應 v25 (Webcam Pro)", layout="wide", page_icon="🥊")

# 顏色定義 (B, G, R)
COLOR_CYAN = (255, 255, 0)    # 左拳提示色 (OpenCV是BGR)
COLOR_RED = (50, 50, 255)     # 右拳提示色
COLOR_TEXT = (255, 255, 255)
COLOR_STROKE = (0, 0, 0)

# 物理常數
SHOULDER_WIDTH_M = 0.45  # 假設一般人肩寬 0.45 公尺 (用於像素轉米)

class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        
        # 狀態機: WAIT_GUARD -> COUNTDOWN -> STIMULUS -> PUNCHING -> RESULT
        self.state = 'WAIT_GUARD'
        self.start_time = 0
        self.stimulus_time = 0
        self.target = None # 'LEFT' or 'RIGHT'
        self.feedback_end_time = 0
        
        # 物理計算變數
        self.prev_landmarks = None
        self.prev_time = 0
        self.max_speed = 0.0
        self.punch_detected_time = 0
        
        # 歷史數據
        self.reaction_history = []
        self.speed_history = []
        self.last_result = {"reaction": 0, "speed": 0, "rating": "", "speed_rating": ""}

        # 字型加載
        self.font_path = "font.ttf"
        self.use_chinese = False
        try:
            ImageFont.truetype(self.font_path, 20)
            self.use_chinese = True
        except:
            print("未找到字型檔，將使用預設字體")

    def put_chinese_text(self, img, text, pos, color, size=30, stroke_width=0):
        """ 使用 PIL 繪製高品質中文 (含描邊) """
        if not self.use_chinese:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size/30, color, 2)
            return img
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(self.font_path, size)
            # 轉換顏色 BGR -> RGB (PIL 使用 RGB)
            pil_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=font, fill=pil_color, stroke_width=stroke_width, stroke_fill=(0,0,0))
        except Exception as e:
            print(f"Font Error: {e}")
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def get_landmarks(self, results, width, height):
        """ 解析關鍵點座標 """
        if not results.pose_landmarks:
            return None
            
        lm = results.pose_landmarks.landmark
        
        # 提取關鍵點 (11,12肩膀; 13,14手肘; 15,16手腕)
        coords = {}
        key_points = {
            'L_SH': 11, 'R_SH': 12,
            'L_EL': 13, 'R_EL': 14,
            'L_WR': 15, 'R_WR': 16,
            'NOSE': 0
        }
        
        for name, idx in key_points.items():
            # x, y 是像素座標, z 是相對深度
            coords[name] = np.array([lm[idx].x * width, lm[idx].y * height, lm[idx].z * width])
            
        return coords

    def calculate_speed(self, current_coords, dt):
        """ 計算拳速 (m/s) """
        if not self.prev_landmarks or dt <= 0:
            return 0.0
            
        # 1. 計算像素比例尺 (Pixels per Meter)
        # 取得當前肩膀像素距離
        shoulder_dist_px = np.linalg.norm(current_coords['L_SH'][:2] - current_coords['R_SH'][:2])
        if shoulder_dist_px < 10: return 0.0 # 避免除以零或雜訊
        
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        # 2. 判斷出拳手
        active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
        
        # 3. 計算手腕位移 (3D距離)
        curr_pos = current_coords[active_wrist]
        prev_pos = self.prev_landmarks[active_wrist]
        dist_px = np.linalg.norm(curr_pos - prev_pos)
        
        # 4. 轉換為真實速度
        speed_mps = (dist_px / pixels_per_meter) / dt
        
        # 過濾雜訊 (人類極限約 15-20 m/s，大於 30 視為誤判)
        if speed_mps > 30: return 0.0
        
        return speed_mps

    def process(self, img):
        # 1. 影像前處理
        img = cv2.flip(img, 1) # 鏡像
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        coords = self.get_landmarks(results, w, h)
        
        # 繪製骨架 (視覺回饋)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # 狀態機邏輯
        if self.state == 'WAIT_GUARD':
            # 顯示指引
            img = self.put_chinese_text(img, "請擺出格鬥姿勢 (雙手舉起)", (20, 50), COLOR_TEXT, 40, stroke_width=2)
            
            if coords:
                # 簡單判定：手腕高於手肘
                l_guard = coords['L_WR'][1] < coords['L_EL'][1]
                r_guard = coords['R_WR'][1] < coords['R_EL'][1]
                
                if l_guard and r_guard:
                    cv2.rectangle(img, (0,0), (w, h), (0, 255, 0), 5) # 綠框提示
                    if current_time - self.start_time > 1.0: # 維持1秒
                        self.state = 'COUNTDOWN'
                        self.start_time = current_time
                else:
                    self.start_time = current_time # 重置計時

        elif self.state == 'COUNTDOWN':
            remaining = 3.0 - (current_time - self.start_time)
            if remaining <= 0:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed = 0
            else:
                # 中央倒數
                cx, cy = int(w/2), int(h/2)
                img = self.put_chinese_text(img, f"{int(remaining)+1}", (cx-20, cy), (0, 255, 255), 100, stroke_width=4)

        elif self.state == 'STIMULUS':
            # 顯示視覺刺激 (v23 風格)
            text = "左拳!" if self.target == 'LEFT' else "右拳!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            
            cx, cy = int(w/2)-100, int(h/2)
            img = self.put_chinese_text(img, text, (cx, cy), color, 120, stroke_width=6)
            
            # 偵測出拳
            if coords and self.prev_landmarks:
                speed = self.calculate_speed(coords, dt)
                if speed > self.max_speed: self.max_speed = speed
                
                # 觸發條件：速度大於閾值 且 手伸直
                # 這裡簡化：只要速度超過 3.5 m/s 且方向正確
                if speed > 3.5:
                    self.state = 'RESULT'
                    reaction_time = (current_time - self.stimulus_time) * 1000
                    self.last_result['reaction'] = reaction_time
                    self.last_result['speed'] = self.max_speed
                    self.feedback_end_time = current_time + 3.0
                    
                    # 記錄數據
                    self.reaction_history.append(reaction_time)
                    self.speed_history.append(self.max_speed)

        elif self.state == 'RESULT':
            # 顯示結果 (v23 評價標準)
            rt = self.last_result['reaction']
            sp = self.last_result['speed']
            
            # 評價邏輯
            if rt < 120: r_txt, r_col = "👑 頂尖", COLOR_CYAN
            elif rt < 250: r_txt, r_col = "🔥 優異", (0, 255, 0)
            else: r_txt, r_col = "😐 一般", (200, 200, 200)
            
            if sp > 13: s_txt, s_col = "💪 職業級", COLOR_RED
            elif sp > 9: s_txt, s_col = "🏆 選手級", (0, 165, 255) # Orange
            else: s_txt, s_col = "🏃 業餘", (255, 255, 0)

            # 繪製結果面板
            overlay = img.copy()
            cv2.rectangle(overlay, (50, h-250), (400, h-50), (0,0,0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            img = self.put_chinese_text(img, f"反應: {rt:.0f} ms", (70, h-200), (255, 255, 255), 30)
            img = self.put_chinese_text(img, f"評價: {r_txt}", (70, h-160), r_col, 30)
            img = self.put_chinese_text(img, f"拳速: {sp:.1f} m/s", (70, h-110), (255, 255, 255), 30)
            img = self.put_chinese_text(img, f"等級: {s_txt}", (70, h-70), s_col, 30)
            
            if current_time > self.feedback_end_time:
                self.state = 'WAIT_GUARD'

        # 更新上一幀座標
        self.prev_landmarks = coords
        return img

# ================= Streamlit 介面 =================
def main():
    st.title("🥊 拳擊反應測試 v25 (Webcam 真人版)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # WebRTC 串流設定
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_streamer(
            key="boxing-pro",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.header("📊 測試數據")
        st.markdown("""
        **使用說明:**
        1. 允許瀏覽器使用鏡頭。
        2. 退後至能看到 **腰部以上** 的位置。
        3. **舉起雙手** (高於手肘) 開始測試。
        4. 看到 **文字提示** 後全力出拳！
        """)
        st.divider()
        st.markdown("### 評價標準")
        st.caption("反應時間: <120ms (頂尖), <250ms (優異)")
        st.caption("出拳速度: >13m/s (職業), >9m/s (選手)")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            # 處理影像並回傳
            processed_img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

if __name__ == "__main__":
    main()
