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
st.set_page_config(page_title="拳擊反應 V28 (高敏度版)", layout="wide", page_icon="🥊")

# 顏色定義 (B, G, R)
COLOR_CYAN = (255, 255, 0)     # 左拳
COLOR_RED = (50, 50, 255)      # 右拳
COLOR_GREEN = (0, 255, 0)      # 成功
COLOR_TEXT = (255, 255, 255)   # 白字
COLOR_WARNING = (0, 165, 255)  # 橘色

# 物理常數
SHOULDER_WIDTH_M = 0.45  # 假設肩寬 0.45 米

class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # 狀態機
        self.state = 'WAIT_GUARD' # WAIT_GUARD -> COUNTDOWN -> STIMULUS -> RESULT -> GAME_OVER
        self.start_time = 0
        self.stimulus_time = 0
        self.target = None 
        self.feedback_end_time = 0
        
        # 流程控制
        self.max_rounds = 10
        self.current_round = 0
        self.is_first_round = True
        
        # 數據記錄
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.last_result = {"reaction": 0, "speed": 0, "hand": ""}
        
        # 物理計算變數
        self.prev_landmarks = None
        self.prev_time = 0
        self.max_speed_in_round = 0.0
        
        # 字型設定 (請確保目錄下有字型檔)
        self.font_path = "font.ttf" 

    def put_chinese_text(self, img, text, pos, color, size=30, stroke_width=0, center_align=False):
        """ 繪製中文文字 (支援置中) """
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype(self.font_path, size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                return img
        
        pil_color = (color[2], color[1], color[0])
        
        # 計算文字大小以便置中
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw_x, draw_y = pos
        if center_align:
            draw_x = pos[0] - text_w // 2
        
        if stroke_width > 0:
            draw.text((draw_x, draw_y), text, font=font, fill=pil_color, stroke_width=stroke_width, stroke_fill=(0,0,0))
        else:
            draw.text((draw_x, draw_y), text, font=font, fill=pil_color)
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def get_landmarks(self, results, width, height):
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        coords = {}
        # 關鍵點: 11(左肩), 12(右肩), 15(左腕), 16(右腕), 13(左肘), 14(右肘)
        key_points = {'L_SH': 11, 'R_SH': 12, 'L_WR': 15, 'R_WR': 16, 'L_EL': 13, 'R_EL': 14}
        
        for name, idx in key_points.items():
            coords[name] = np.array([lm[idx].x * width, lm[idx].y * height])
        return coords

    def calculate_elbow_angle(self, sh, el, wr):
        """ 計算手肘角度 (0~180度) """
        # 向量 A: 手肘->肩膀, 向量 B: 手肘->手腕
        vec_a = sh - el
        vec_b = wr - el
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        cosine_angle = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def detect_punch_v2(self, coords, dt):
        """ 
        V2 高敏度判斷：
        結合「手腕延伸速度」與「手肘移動速度」
        """
        if not self.prev_landmarks or dt <= 0:
            return 0.0, False
            
        # 1. 像素轉公尺比例
        shoulder_dist_px = np.linalg.norm(coords['L_SH'] - coords['R_SH'])
        if shoulder_dist_px < 10: return 0.0, False
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        # 2. 鎖定目標手
        target_hand = 'LEFT' if self.target == 'LEFT' else 'RIGHT'
        sh_key = 'L_SH' if target_hand == 'LEFT' else 'R_SH'
        el_key = 'L_EL' if target_hand == 'LEFT' else 'R_EL'
        wr_key = 'L_WR' if target_hand == 'LEFT' else 'R_WR'
        
        # 3. 計算各部位速度 (相對於上一幀的位移)
        # 手腕速度
        wrist_disp = np.linalg.norm(coords[wr_key] - self.prev_landmarks[wr_key])
        wrist_speed = (wrist_disp / pixels_per_meter) / dt
        
        # 手肘速度 (新增參照：肘部快速移動通常代表出拳起始)
        elbow_disp = np.linalg.norm(coords[el_key] - self.prev_landmarks[el_key])
        elbow_speed = (elbow_disp / pixels_per_meter) / dt
        
        # 4. 計算手臂延伸狀態
        curr_arm_len = np.linalg.norm(coords[sh_key] - coords[wr_key])
        prev_arm_len = np.linalg.norm(self.prev_landmarks[sh_key] - self.prev_landmarks[wr_key])
        is_extending = curr_arm_len > prev_arm_len # 手臂正在變長
        
        # 5. 綜合速度 (給予手腕 70% 權重, 手肘 30% 權重)
        # 這樣即使手腕被遮擋或模糊，手肘的快速移動也能幫助觸發
        composite_speed = (wrist_speed * 0.7) + (elbow_speed * 0.3)
        
        # 6. 觸發判定
        # 條件 A: 綜合速度 > 1.5 m/s (門檻降低) 且 手臂正在伸長
        # 條件 B: 手腕瞬間極快 (> 3.0 m/s) (防誤判備用)
        is_punch = False
        if (composite_speed > 1.5 and is_extending) or (wrist_speed > 3.0):
            is_punch = True
            
        return composite_speed, is_punch

    def reset_game(self):
        self.state = 'WAIT_GUARD'
        self.current_round = 0
        self.is_first_round = True
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.prev_landmarks = None

    def process(self, img):
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        coords = self.get_landmarks(results, w, h)
        
        # 顯示骨架輔助
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # ================= 狀態機 =================
        
        if self.state == 'GAME_OVER':
            # 遮罩
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)
            
            # 數據計算
            l_rt = np.mean(self.left_stats['reaction']) if self.left_stats['reaction'] else 0
            l_sp = np.mean(self.left_stats['speed']) if self.left_stats['speed'] else 0
            r_rt = np.mean(self.right_stats['reaction']) if self.right_stats['reaction'] else 0
            r_sp = np.mean(self.right_stats['speed']) if self.right_stats['speed'] else 0
            total_avg_rt = (l_rt + r_rt) / 2 if (l_rt+r_rt) > 0 else 0
            
            # 評級
            rank = "C"
            rank_color = (200, 200, 200)
            if total_avg_rt > 0:
                if total_avg_rt < 250: rank, rank_color = "S (神速)", COLOR_CYAN
                elif total_avg_rt < 350: rank, rank_color = "A (優秀)", COLOR_GREEN
                elif total_avg_rt < 450: rank, rank_color = "B (普通)", COLOR_WARNING
            
            # === UI 重構：防止文字重疊 ===
            cx = int(w/2)
            
            # 1. 標題 (高度 10%)
            img = self.put_chinese_text(img, "=== 測驗結果 ===", (cx, int(h*0.1)), COLOR_TEXT, 50, 2, center_align=True)
            
            # 2. 數據欄位 (左欄 & 右欄) (高度 25% 開始)
            col_y_start = int(h * 0.25)
            line_gap = 50
            
            # 左手 (左側 1/4 處)
            lx = int(w * 0.25)
            img = self.put_chinese_text(img, "【左手】", (lx, col_y_start), COLOR_CYAN, 40, 2, center_align=True)
            img = self.put_chinese_text(img, f"反應: {l_rt:.0f} ms", (lx, col_y_start + line_gap), COLOR_TEXT, 30, center_align=True)
            img = self.put_chinese_text(img, f"均速: {l_sp:.1f} m/s", (lx, col_y_start + line_gap*2), COLOR_TEXT, 30, center_align=True)
            
            # 右手 (右側 3/4 處)
            rx = int(w * 0.75)
            img = self.put_chinese_text(img, "【右手】", (rx, col_y_start), COLOR_RED, 40, 2, center_align=True)
            img = self.put_chinese_text(img, f"反應: {r_rt:.0f} ms", (rx, col_y_start + line_gap), COLOR_TEXT, 30, center_align=True)
            img = self.put_chinese_text(img, f"均速: {r_sp:.1f} m/s", (rx, col_y_start + line_gap*2), COLOR_TEXT, 30, center_align=True)
            
            # 3. 總評 (高度 60% 處)
            img = self.put_chinese_text(img, f"綜合等級: {rank}", (cx, int(h*0.6)), rank_color, 60, 3, center_align=True)
            
            # 4. 重試按鈕 (底部)
            btn_w, btn_h = 240, 80
            btn_x1, btn_y1 = cx - btn_w//2, int(h * 0.8)
            btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h
            
            cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 255), 3)
            img = self.put_chinese_text(img, "重新測驗", (cx, btn_y1+20), (0, 255, 255), 40, center_align=True)

            # 檢測手部觸發重置
            if coords:
                for hand in ['L_WR', 'R_WR']:
                    hx, hy = coords[hand]
                    if btn_x1 < hx < btn_x2 and btn_y1 < hy < btn_y2:
                        cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), COLOR_GREEN, -1)
                        self.reset_game()

        elif self.state == 'WAIT_GUARD':
            hold_time = 3.0 if self.is_first_round else 2.0
            
            # 顯示回合
            img = self.put_chinese_text(img, f"Round {self.current_round + 1}/10", (30, 50), COLOR_TEXT, 40, 2)
            
            # 顯示倒數
            elapsed = current_time - self.start_time
            remain = max(0.0, hold_time - elapsed)
            
            # 中央提示
            cx, cy = int(w/2), int(h/2)
            if coords:
                # 簡單判定防禦：手腕在肩膀以上 (稍微放寬判定，避免一直重置)
                l_guard = coords['L_WR'][1] < coords['L_SH'][1] + 50
                r_guard = coords['R_WR'][1] < coords['R_SH'][1] + 50
                
                if l_guard and r_guard:
                    # 進度條
                    bar_len = 300
                    prog = min(elapsed / hold_time, 1.0)
                    cv2.rectangle(img, (cx - bar_len//2, cy+80), (cx - bar_len//2 + int(bar_len*prog), cy+100), COLOR_GREEN, -1)
                    cv2.rectangle(img, (cx - bar_len//2, cy+80), (cx + bar_len//2, cy+100), COLOR_TEXT, 2)
                    
                    img = self.put_chinese_text(img, f"保持防禦... {remain:.1f}", (cx, cy), COLOR_GREEN, 40, 2, center_align=True)
                    
                    if elapsed >= hold_time:
                        self.state = 'COUNTDOWN'
                        self.start_time = current_time
                else:
                    self.start_time = current_time
                    img = self.put_chinese_text(img, "請舉起雙手", (cx, cy), COLOR_WARNING, 50, 2, center_align=True)
            else:
                 img = self.put_chinese_text(img, "偵測不到人像", (cx, cy), COLOR_RED, 40, 2, center_align=True)

        elif self.state == 'COUNTDOWN':
            # 隨機延遲 1~2.5秒
            delay = random.uniform(1.0, 2.5)
            if current_time - self.start_time > delay:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed_in_round = 0
            else:
                # 專注點
                cv2.circle(img, (int(w/2), int(h/2)), 25, (255, 255, 255), -1)

        elif self.state == 'STIMULUS':
            # 視覺刺激
            text = "左拳!" if self.target == 'LEFT' else "右拳!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            img = self.put_chinese_text(img, text, (int(w/2), int(h/2)-50), color, 120, 5, center_align=True)
            
            if coords:
                speed, is_punch = self.detect_punch_v2(coords, dt)
                
                if speed > self.max_speed_in_round:
                    self.max_speed_in_round = speed
                
                # 判定成功
                if is_punch:
                    rt = (current_time - self.stimulus_time) * 1000
                    if rt > 80: # 過濾 <80ms 的異常數據
                        self.last_result = {
                            "reaction": rt,
                            "speed": self.max_speed_in_round,
                            "hand": self.target
                        }
                        # 記錄
                        if self.target == 'LEFT':
                            self.left_stats['reaction'].append(rt)
                            self.left_stats['speed'].append(self.max_speed_in_round)
                        else:
                            self.right_stats['reaction'].append(rt)
                            self.right_stats['speed'].append(self.max_speed_in_round)
                            
                        self.state = 'RESULT'
                        self.feedback_end_time = current_time + 1.2
                        self.current_round += 1
                        self.is_first_round = False

        elif self.state == 'RESULT':
            res = self.last_result
            color = COLOR_CYAN if res['hand'] == 'LEFT' else COLOR_RED
            cx, cy = int(w/2), int(h/2)
            
            img = self.put_chinese_text(img, f"{res['reaction']:.0f} ms", (cx, cy-60), color, 80, 3, center_align=True)
            img = self.put_chinese_text(img, f"{res['speed']:.1f} m/s", (cx, cy+50), COLOR_TEXT, 50, 2, center_align=True)
            
            if current_time > self.feedback_end_time:
                if self.current_round >= self.max_rounds:
                    self.state = 'GAME_OVER'
                else:
                    self.state = 'WAIT_GUARD'
                    self.start_time = current_time

        self.prev_landmarks = coords
        return img

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            processed = self.logic.process(img)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")
        except Exception:
            return frame

def main():
    st.title("🥊 拳擊反應測試 V28 (高敏度版)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        webrtc_streamer(
            key="boxing-v28",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col2:
        st.markdown("### 🛠️ V28 更新說明")
        st.markdown("""
        1. **肘關節感應**: 加入手肘移動偵測，出拳判斷更靈敏。
        2. **介面優化**: 修正結果畫面文字重疊問題。
        
        **使用注意:**
        * 請確保資料夾內有 `font.ttf` 以顯示中文。
        * 保持環境光線充足。
        """)

if __name__ == "__main__":
    main()
