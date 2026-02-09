import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import mediapipe as mp
import time
import random
from PIL import ImageFont, ImageDraw, Image

# ================= 配置與常數 =================
st.set_page_config(page_title="拳擊反應 v26 (Pro)", layout="wide", page_icon="🥊")

# 顏色定義 (B, G, R)
COLOR_CYAN = (255, 255, 0)
COLOR_RED = (50, 50, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG_DARK = (0, 0, 0)

# 物理常數
SHOULDER_WIDTH_M = 0.45  # 假設一般人肩寬 0.45 公尺
SMOOTHING_FACTOR = 0.5   # 關鍵點平滑係數 (0~1, 越小越平滑但延遲越高)

class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7, # 提高信心度以減少雜訊
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # 遊戲狀態變數
        self.state = 'WAIT_GUARD' # WAIT_GUARD -> COUNTDOWN -> STIMULUS -> RESULT -> GAME_OVER
        self.start_time = 0
        self.stimulus_time = 0
        self.target = None 
        self.feedback_end_time = 0
        
        # 測驗流程控制
        self.max_rounds = 10
        self.current_round = 0
        self.is_first_round = True
        
        # 數據記錄
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.last_result = {"reaction": 0, "speed": 0, "hand": ""}
        
        # 物理計算變數 (用於濾波)
        self.prev_landmarks_smooth = None
        self.prev_time = 0
        self.max_speed_in_round = 0.0
        
        # 字型
        self.font_path = "arial.ttf" # 預設 fallback
        # 嘗試尋找系統中文字型 (Linux/Windows/Mac 路徑可能不同，這裡僅做簡單處理)
        self.use_chinese = False
        # 為了演示方便，若沒有中文字型檔，會退回 OpenCV 繪圖

    def put_text_pil(self, img, text, pos, color, size=30, stroke=0, bg_color=None):
        """ 使用 PIL 繪製文字 (支援中文，需自行上傳字型檔，否則使用 cv2) """
        # 這裡簡化處理：如果這只是 Demo，我們用 OpenCV 繪製英文或簡單中文
        # 若需要漂亮中文，請確保環境有 .ttf 檔案
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            # 嘗試加載字型 (請確保目錄下有字型檔，例如 msjh.ttc 或 simhei.ttf)
            # 這裡為了通用性，若失敗則用預設
            font = ImageFont.truetype("font.ttf", size) 
        except:
            font = ImageFont.load_default()
            
        pil_color = (color[2], color[1], color[0])
        
        # 繪製背景框 (如果有的話)
        if bg_color:
            text_bbox = draw.textbbox(pos, text, font=font)
            # 擴大一點背景
            bg_box = (text_bbox[0]-10, text_bbox[1]-10, text_bbox[2]+10, text_bbox[3]+10)
            draw.rectangle(bg_box, fill=(bg_color[2], bg_color[1], bg_color[0]))

        draw.text(pos, text, font=font, fill=pil_color, stroke_width=stroke, stroke_fill=(0,0,0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def smooth_landmarks(self, current_coords):
        """ 簡單的指數平滑濾波，減少抖動 """
        if self.prev_landmarks_smooth is None:
            self.prev_landmarks_smooth = current_coords
            return current_coords
            
        smoothed = {}
        for key, val in current_coords.items():
            prev = self.prev_landmarks_smooth[key]
            # 公式: new = alpha * curr + (1 - alpha) * prev
            smoothed[key] = SMOOTHING_FACTOR * val + (1 - SMOOTHING_FACTOR) * prev
            
        self.prev_landmarks_smooth = smoothed
        return smoothed

    def get_landmarks(self, results, width, height):
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        coords = {}
        # 關鍵點: 11(左肩), 12(右肩), 15(左腕), 16(右腕), 13(左肘), 14(右肘)
        key_points = {'L_SH': 11, 'R_SH': 12, 'L_WR': 15, 'R_WR': 16, 'L_EL': 13, 'R_EL': 14, 'NOSE': 0}
        
        for name, idx in key_points.items():
            # 包含 Z 軸 (深度)
            coords[name] = np.array([lm[idx].x * width, lm[idx].y * height, lm[idx].z * width])
        return coords

    def calculate_speed(self, current_coords, dt):
        """ 優化後的速度計算 """
        if not self.prev_landmarks_smooth or dt <= 0:
            return 0.0
            
        # 1. 動態計算像素比例尺 (每幀都算，避免人前後移動導致誤差)
        shoulder_dist_px = np.linalg.norm(current_coords['L_SH'][:2] - current_coords['R_SH'][:2])
        if shoulder_dist_px < 10: return 0.0
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        # 2. 鎖定目標手
        active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
        
        # 3. 計算位移 (使用平滑後的座標)
        curr_pos = current_coords[active_wrist]
        prev_pos = self.prev_landmarks_smooth[active_wrist]
        
        # 計算 3D 距離，但降低 Z 軸權重 (因為 webcam 的深度估計雜訊最大)
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        dz = (curr_pos[2] - prev_pos[2]) * 0.5 # 降低 Z 軸影響
        
        dist_px = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 4. 換算速度
        speed_mps = (dist_px / pixels_per_meter) / dt
        
        # 5. 過濾異常值 (物理極限過濾)
        if speed_mps > 25: return 0.0 # 超過 25m/s 通常是誤判
        
        return speed_mps

    def reset_game(self):
        self.state = 'WAIT_GUARD'
        self.current_round = 0
        self.is_first_round = True
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.prev_landmarks_smooth = None

    def process(self, img):
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # 取得並平滑座標
        raw_coords = self.get_landmarks(results, w, h)
        coords = None
        if raw_coords:
            coords = self.smooth_landmarks(raw_coords)
            # 繪製骨架
            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # ================= 狀態機 =================
        
        if self.state == 'GAME_OVER':
            # 繪製半透明黑色遮罩
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
            
            # 計算平均數據
            l_n = len(self.left_stats['reaction'])
            r_n = len(self.right_stats['reaction'])
            
            l_rt = np.mean(self.left_stats['reaction']) if l_n > 0 else 0
            l_sp = np.mean(self.left_stats['speed']) if l_n > 0 else 0
            r_rt = np.mean(self.right_stats['reaction']) if r_n > 0 else 0
            r_sp = np.mean(self.right_stats['speed']) if r_n > 0 else 0
            
            total_avg_rt = (l_rt + r_rt) / 2 if (l_n+r_n) > 0 else 0
            
            # 總評級
            rank = "C"
            if total_avg_rt > 0:
                if total_avg_rt < 250: rank = "S (神級)"
                elif total_avg_rt < 300: rank = "A (職業)"
                elif total_avg_rt < 400: rank = "B (一般)"
                else: rank = "C (加油)"

            # 顯示報告
            cy = int(h/2)
            cv2.putText(img, "TEST COMPLETE", (int(w/2)-150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 3)
            
            # 左手數據
            cv2.putText(img, "LEFT HAND", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CYAN, 2)
            cv2.putText(img, f"Reaction: {l_rt:.0f} ms", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
            cv2.putText(img, f"Speed: {l_sp:.1f} m/s", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

            # 右手數據
            cv2.putText(img, "RIGHT HAND", (w-350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
            cv2.putText(img, f"Reaction: {r_rt:.0f} ms", (w-350, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
            cv2.putText(img, f"Speed: {r_sp:.1f} m/s", (w-350, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

            # 總評
            cv2.putText(img, f"RANK: {rank}", (int(w/2)-120, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GREEN, 3)

            # 繪製虛擬按鈕區域
            btn_w, btn_h = 200, 80
            btn_x1, btn_y1 = int(w/2) - btn_w//2, h - 150
            btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h
            
            cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 255), 2)
            cv2.putText(img, "RETRY", (btn_x1+40, btn_y1+55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            
            cv2.putText(img, "Put hand here to Retry", (btn_x1-20, btn_y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # 檢測手是否在按鈕區域內 (重置遊戲)
            if coords:
                l_hand = coords['L_WR']
                r_hand = coords['R_WR']
                # 簡單判定：只要有一隻手進入方框
                if (btn_x1 < l_hand[0] < btn_x2 and btn_y1 < l_hand[1] < btn_y2) or \
                   (btn_x1 < r_hand[0] < btn_x2 and btn_y1 < r_hand[1] < btn_y2):
                       cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 0), -1) # 變綠色表示觸發
                       self.reset_game()

        elif self.state == 'WAIT_GUARD':
            # 計算需要的保持時間
            hold_time_needed = 3.0 if self.is_first_round else 2.0
            
            msg = f"Round {self.current_round + 1} / {self.max_rounds}"
            cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
            
            instruction = f"GUARD UP ({hold_time_needed}s)"
            cv2.putText(img, instruction, (int(w/2)-150, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2)

            if coords:
                # 判定防禦姿勢 (手腕高於手肘)
                l_guard = coords['L_WR'][1] < coords['L_EL'][1]
                r_guard = coords['R_WR'][1] < coords['R_EL'][1]
                
                if l_guard and r_guard:
                    # 進度條視覺化
                    elapsed = current_time - self.start_time
                    progress = min(elapsed / hold_time_needed, 1.0)
                    bar_w = 400
                    cv2.rectangle(img, (int(w/2)-200, int(h/2)+50), (int(w/2)-200 + int(bar_w*progress), int(h/2)+70), (0, 255, 0), -1)
                    
                    if elapsed >= hold_time_needed:
                        self.state = 'COUNTDOWN'
                        self.start_time = current_time
                else:
                    self.start_time = current_time # 姿勢不對，重置計時

        elif self.state == 'COUNTDOWN':
            # 隨機倒數 1~3 秒之間讓刺激更不可預測
            countdown_dur = 1.0 
            remaining = countdown_dur - (current_time - self.start_time)
            
            if remaining <= 0:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed_in_round = 0
            else:
                cv2.circle(img, (int(w/2), int(h/2)), 50, (255, 255, 255), 2)
                # 這裡不顯示數字，改用專注的圓點，模擬真實訓練

        elif self.state == 'STIMULUS':
            # 顯示視覺訊號
            text = "LEFT!" if self.target == 'LEFT' else "RIGHT!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            # 大字提示
            cv2.putText(img, text, (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 8)
            
            if coords:
                # 計算當前速度
                speed = self.calculate_speed(coords, dt)
                if speed > self.max_speed_in_round:
                    self.max_speed_in_round = speed
                
                # 觸發判定: 速度足夠快 且 使用正確的手
                is_correct_hand = False
                if self.target == 'LEFT' and coords['L_WR'][1] < h/2: # 簡單判定：手要舉起來打
                     if speed > 2.0: is_correct_hand = True # 閾值 2.0 m/s
                elif self.target == 'RIGHT' and coords['R_WR'][1] < h/2:
                     if speed > 2.0: is_correct_hand = True

                if is_correct_hand and speed > 3.0: # 確定的出拳
                    reaction_time = (current_time - self.stimulus_time) * 1000
                    
                    # 記錄本次結果
                    self.last_result = {
                        "reaction": reaction_time,
                        "speed": self.max_speed_in_round,
                        "hand": self.target
                    }
                    
                    # 存入歷史
                    if self.target == 'LEFT':
                        self.left_stats['reaction'].append(reaction_time)
                        self.left_stats['speed'].append(self.max_speed_in_round)
                    else:
                        self.right_stats['reaction'].append(reaction_time)
                        self.right_stats['speed'].append(self.max_speed_in_round)

                    # 狀態切換
                    self.state = 'RESULT'
                    self.feedback_end_time = current_time + 1.5 # 顯示結果 1.5 秒
                    self.is_first_round = False # 第一局結束
                    self.current_round += 1

        elif self.state == 'RESULT':
            # 顯示當下這拳的數據
            res = self.last_result
            color = COLOR_CYAN if res['hand'] == 'LEFT' else COLOR_RED
            
            cv2.putText(img, f"{res['reaction']:.0f} ms", (int(w/2)-80, int(h/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            cv2.putText(img, f"{res['speed']:.1f} m/s", (int(w/2)-80, int(h/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
            
            if current_time > self.feedback_end_time:
                if self.current_round >= self.max_rounds:
                    self.state = 'GAME_OVER'
                else:
                    self.state = 'WAIT_GUARD'
                    self.start_time = current_time # 重置計時給 Guard

        return img

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            processed_img = self.logic.process(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

def main():
    st.title("🥊 專業拳擊反應測試 v26")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("請允許攝影機權限。測驗共 10 回合。")
        webrtc_streamer(
            key="boxing-pro-v26",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.markdown("### 📜 規則說明")
        st.markdown("""
        1. **準備姿勢**：雙手舉高 (護頭)。
        2. **啟動時間**：第一下需維持 **3秒**，之後每下維持 **2秒**。
        3. **視覺訊號**：看到 **LEFT** 或 **RIGHT** 立即出拳。
        4. **結算**：10下後顯示平均成績。
        5. **重來**：在結算畫面，將手放在黃色框框內即可重測。
        """)
        
        st.markdown("---")
        st.markdown("### 📊 評價標準")
        st.caption("⚡ 反應時間 (ms)")
        st.text("S級: < 250ms")
        st.text("A級: < 300ms")
        
        st.caption("🚀 拳速 (m/s)")
        st.text("職業級: > 10.0 m/s")
        st.text("一般人: 5.0 - 8.0 m/s")

if __name__ == "__main__":
    main()
