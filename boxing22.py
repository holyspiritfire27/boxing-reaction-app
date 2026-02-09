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
st.set_page_config(page_title="拳擊反應 v27 (中文修正版)", layout="wide", page_icon="🥊")

# 顏色定義 (B, G, R)
COLOR_CYAN = (255, 255, 0)     # 左拳提示
COLOR_RED = (50, 50, 255)      # 右拳提示
COLOR_GREEN = (0, 255, 0)      # 成功/良好
COLOR_TEXT = (255, 255, 255)   # 白字
COLOR_WARNING = (0, 165, 255)  # 橘色

# 物理常數
SHOULDER_WIDTH_M = 0.45  # 假設肩寬 0.45 米

class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 初始化 (提高偵測靈敏度)
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
        
        # 測驗流程控制
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
        self.prev_arm_len = {'LEFT': 0, 'RIGHT': 0} # 記錄上一幀手臂長度
        self.max_speed_in_round = 0.0
        
        # 字型設定
        self.font_path = "font.ttf" # 請確保有此檔案，或更改為系統字型路徑
        self.use_chinese = True

    def put_chinese_text(self, img, text, pos, color, size=30, stroke_width=0, bg=False):
        """ 繪製中文文字 """
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype(self.font_path, size)
        except:
            # 如果找不到字型，嘗試載入系統預設，或退回英文模式
            try:
                font = ImageFont.load_default()
            except:
                return img # 放棄繪製
        
        pil_color = (color[2], color[1], color[0])
        
        if bg:
            # 簡單繪製文字背景框
            bbox = draw.textbbox(pos, text, font=font)
            expand = 5
            draw.rectangle((bbox[0]-expand, bbox[1]-expand, bbox[2]+expand, bbox[3]+expand), fill=(0,0,0))

        if stroke_width > 0:
            draw.text(pos, text, font=font, fill=pil_color, stroke_width=stroke_width, stroke_fill=(0,0,0))
        else:
            draw.text(pos, text, font=font, fill=pil_color)
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def get_landmarks(self, results, width, height):
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        coords = {}
        # 關鍵點: 11(左肩), 12(右肩), 15(左腕), 16(右腕), 13(左肘), 14(右肘)
        key_points = {'L_SH': 11, 'R_SH': 12, 'L_WR': 15, 'R_WR': 16, 'L_EL': 13, 'R_EL': 14}
        
        for name, idx in key_points.items():
            # 這裡我們只取 x, y 進行 2D 投影計算，Z 軸在 Webcam 雜訊太大暫不使用
            coords[name] = np.array([lm[idx].x * width, lm[idx].y * height])
        return coords

    def calculate_metrics(self, coords, dt):
        """ 
        改進版物理計算：
        使用「手臂伸展速度」而非單純的手腕移動速度。
        這樣可以避免身體前後晃動造成的誤判。
        """
        if not self.prev_landmarks or dt <= 0:
            return 0.0, False
            
        # 1. 計算像素比例尺 (Pixels per Meter)
        shoulder_dist_px = np.linalg.norm(coords['L_SH'] - coords['R_SH'])
        if shoulder_dist_px < 10: return 0.0, False
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        # 2. 鎖定目標手
        target_hand = 'LEFT' if self.target == 'LEFT' else 'RIGHT'
        sh_key = 'L_SH' if target_hand == 'LEFT' else 'R_SH'
        wr_key = 'L_WR' if target_hand == 'LEFT' else 'R_WR'
        
        # 3. 計算「肩膀-手腕」的距離 (手臂延伸長度)
        curr_arm_len = np.linalg.norm(coords[sh_key] - coords[wr_key])
        prev_arm_len = self.prev_arm_len.get(target_hand, curr_arm_len)
        
        # 4. 計算延伸速度 (Extension Velocity)
        # 只有當手臂「變長」(伸出去) 時才計算正速度
        delta_len = curr_arm_len - prev_arm_len
        if delta_len > 0:
            speed_mps = (delta_len / pixels_per_meter) / dt
        else:
            speed_mps = 0
            
        # 更新記錄
        self.prev_arm_len[target_hand] = curr_arm_len
        
        # 5. 判定是否為有效出拳 (速度夠快 且 真的有伸出去)
        # 門檻設為 2.0 m/s 比較容易觸發
        is_punch = False
        if speed_mps > 2.0:
            is_punch = True
            
        return speed_mps, is_punch

    def reset_game(self):
        self.state = 'WAIT_GUARD'
        self.current_round = 0
        self.is_first_round = True
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.prev_landmarks = None

    def process(self, img):
        # 影像前處理
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        coords = self.get_landmarks(results, w, h)
        
        # 繪製骨架
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # ================= 狀態機邏輯 =================
        
        if self.state == 'GAME_OVER':
            # 黑色半透明遮罩
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
            
            # 計算數據
            l_rt = np.mean(self.left_stats['reaction']) if self.left_stats['reaction'] else 0
            l_sp = np.mean(self.left_stats['speed']) if self.left_stats['speed'] else 0
            r_rt = np.mean(self.right_stats['reaction']) if self.right_stats['reaction'] else 0
            r_sp = np.mean(self.right_stats['speed']) if self.right_stats['speed'] else 0
            
            total_avg_rt = (l_rt + r_rt) / 2 if (l_rt+r_rt) > 0 else 0
            
            # 總評級
            rank = "C (加油)"
            rank_color = (200, 200, 200)
            if total_avg_rt > 0:
                if total_avg_rt < 250: 
                    rank = "S (神級)"
                    rank_color = COLOR_CYAN
                elif total_avg_rt < 300: 
                    rank = "A (職業)"
                    rank_color = COLOR_GREEN
                elif total_avg_rt < 400: 
                    rank = "B (一般)"
                    rank_color = COLOR_WARNING

            # 顯示報告 (中文)
            cx = int(w/2)
            img = self.put_chinese_text(img, "測驗結束", (cx-100, 60), COLOR_TEXT, 50, 2)
            
            # 左手數據
            img = self.put_chinese_text(img, "左手數據", (100, 150), COLOR_CYAN, 40, 2)
            img = self.put_chinese_text(img, f"反應: {l_rt:.0f} ms", (100, 200), COLOR_TEXT, 30)
            img = self.put_chinese_text(img, f"均速: {l_sp:.1f} m/s", (100, 250), COLOR_TEXT, 30)

            # 右手數據
            img = self.put_chinese_text(img, "右手數據", (w-300, 150), COLOR_RED, 40, 2)
            img = self.put_chinese_text(img, f"反應: {r_rt:.0f} ms", (w-300, 200), COLOR_TEXT, 30)
            img = self.put_chinese_text(img, f"均速: {r_sp:.1f} m/s", (w-300, 250), COLOR_TEXT, 30)

            # 總評
            img = self.put_chinese_text(img, f"綜合等級: {rank}", (cx-150, h//2 + 50), rank_color, 45, 2)

            # 重試按鈕區域
            btn_x1, btn_y1 = cx - 100, h - 150
            btn_x2, btn_y2 = cx + 100, h - 70
            
            cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 255), 2)
            img = self.put_chinese_text(img, "重新測驗", (btn_x1+25, btn_y1+15), (0, 255, 255), 40)
            img = self.put_chinese_text(img, "將手放入框內以重置", (btn_x1-30, btn_y1-30), (200,200,200), 20)

            # 檢測手是否觸發按鈕
            if coords:
                for hand in ['L_WR', 'R_WR']:
                    hx, hy = coords[hand]
                    if btn_x1 < hx < btn_x2 and btn_y1 < hy < btn_y2:
                        cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 0), -1)
                        self.reset_game()

        elif self.state == 'WAIT_GUARD':
            # 準備階段：顯示回合數與指示
            hold_time_needed = 3.0 if self.is_first_round else 2.0
            
            msg = f"第 {self.current_round + 1} 回合 / 共 10 回"
            img = self.put_chinese_text(img, msg, (20, 50), COLOR_TEXT, 30, 2)
            
            instruction = f"雙手舉高 ({hold_time_needed}秒)"
            img = self.put_chinese_text(img, instruction, (int(w/2)-120, int(h/2)), COLOR_TEXT, 40, 2)

            if coords:
                # 判定防禦姿勢 (手腕 y < 手肘 y)
                l_guard = coords['L_WR'][1] < coords['L_EL'][1]
                r_guard = coords['R_WR'][1] < coords['R_EL'][1]
                
                if l_guard and r_guard:
                    # 綠色進度條
                    elapsed = current_time - self.start_time
                    progress = min(elapsed / hold_time_needed, 1.0)
                    bar_w = 400
                    cx = int(w/2)
                    cv2.rectangle(img, (cx-200, int(h/2)+60), (cx-200 + int(bar_w*progress), int(h/2)+80), COLOR_GREEN, -1)
                    
                    if elapsed >= hold_time_needed:
                        self.state = 'COUNTDOWN'
                        self.start_time = current_time
                else:
                    self.start_time = current_time # 姿勢不對，重置

        elif self.state == 'COUNTDOWN':
            # 隨機延遲
            delay = random.uniform(1.0, 2.5) # 1~2.5秒隨機
            if current_time - self.start_time > delay:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed_in_round = 0
                # 重置上一幀手臂長度，避免瞬間誤差
                if coords:
                    self.prev_arm_len['LEFT'] = np.linalg.norm(coords['L_SH'] - coords['L_WR'])
                    self.prev_arm_len['RIGHT'] = np.linalg.norm(coords['R_SH'] - coords['R_WR'])
            else:
                # 顯示準備圓點
                cv2.circle(img, (int(w/2), int(h/2)), 20, (255, 255, 255), -1)

        elif self.state == 'STIMULUS':
            # 視覺刺激
            text = "左拳!" if self.target == 'LEFT' else "右拳!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            img = self.put_chinese_text(img, text, (int(w/2)-80, int(h/2)-50), color, 100, 5)
            
            if coords:
                speed, is_punch = self.calculate_metrics(coords, dt)
                
                # 記錄最大速度
                if speed > self.max_speed_in_round:
                    self.max_speed_in_round = speed
                
                # 判定出拳
                # 條件：偵測到出拳動作(is_punch) 且 目標手正確
                if is_punch:
                    reaction_time = (current_time - self.stimulus_time) * 1000
                    
                    # 簡單過濾太短的反應時間 (避免預判或雜訊)
                    if reaction_time > 100: 
                        self.last_result = {
                            "reaction": reaction_time,
                            "speed": self.max_speed_in_round,
                            "hand": self.target
                        }
                        
                        # 存檔
                        if self.target == 'LEFT':
                            self.left_stats['reaction'].append(reaction_time)
                            self.left_stats['speed'].append(self.max_speed_in_round)
                        else:
                            self.right_stats['reaction'].append(reaction_time)
                            self.right_stats['speed'].append(self.max_speed_in_round)

                        self.state = 'RESULT'
                        self.feedback_end_time = current_time + 1.5
                        self.is_first_round = False
                        self.current_round += 1

        elif self.state == 'RESULT':
            # 顯示單次結果
            res = self.last_result
            color = COLOR_CYAN if res['hand'] == 'LEFT' else COLOR_RED
            
            img = self.put_chinese_text(img, f"{res['reaction']:.0f} ms", (int(w/2)-100, int(h/2)-60), color, 60, 3)
            img = self.put_chinese_text(img, f"速度: {res['speed']:.1f} m/s", (int(w/2)-100, int(h/2)+40), COLOR_TEXT, 40, 2)
            
            if current_time > self.feedback_end_time:
                if self.current_round >= self.max_rounds:
                    self.state = 'GAME_OVER'
                else:
                    self.state = 'WAIT_GUARD'
                    self.start_time = current_time

        # 更新上一幀
        self.prev_landmarks = coords
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
    st.title("🥊 拳擊反應測試 v27 (中文專業版)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        webrtc_streamer(
            key="boxing-pro-v27",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.header("📊 說明")
        st.markdown("""
        **操作指南:**
        1. 確保有上傳 `font.ttf` (字型檔)。
        2. 站在鏡頭前，露出上半身。
        3. **雙手舉高** (高於手肘) 觸發開始。
        4. 看到 **左拳/右拳** 提示，全力出拳！
        
        **規則:**
        * 首局預備 3 秒，之後 2 秒。
        * 共 10 回合。
        * 結束後顯示詳細數據與評級。
        """)
        st.divider()
        st.info("💡 提示：出拳時請將手臂**完全伸直**，系統更容易偵測到速度。")

if __name__ == "__main__":
    main()
