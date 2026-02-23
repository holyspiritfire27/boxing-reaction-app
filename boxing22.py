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
st.set_page_config(page_title="拳擊反應 V30 (命中判定版)", layout="wide", page_icon="🥊")

# 顏色定義 (B, G, R)
COLOR_CYAN = (255, 255, 0)     # 左拳
COLOR_RED = (50, 50, 255)      # 右拳
COLOR_GREEN = (0, 255, 0)      # 成功/命中
COLOR_ERROR = (0, 0, 255)      # 失誤 (紅色)
COLOR_TEXT = (255, 255, 255)   # 白字
COLOR_WARNING = (0, 165, 255)  # 橘色

# 物理常數
SHOULDER_WIDTH_M = 0.45  # 假設肩寬 0.45 米

class BoxingAnalystLogic:
    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, # 進一步降低門檻，提升敏感度
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # 狀態機
        self.state = 'WAIT_GUARD' 
        self.start_time = 0
        self.stimulus_time = 0
        self.target = None 
        self.game_over_time = 0
        
        # 流程控制
        self.max_rounds = 10
        self.current_round = 0
        self.is_first_round = True
        
        # 數據記錄
        self.hit_count = 0  # 記錄命中次數
        self.left_stats = {'reaction': [], 'speed': []}
        self.right_stats = {'reaction': [], 'speed': []}
        self.last_result = {"reaction": 0, "speed": 0, "target": "", "punched_hand": "", "is_hit": False}
        
        # 物理計算變數
        self.prev_landmarks = None
        self.prev_time = 0
        self.max_speed_in_round = 0.0
        
        # 字型設定
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
        key_points = {'L_SH': 11, 'R_SH': 12, 'L_WR': 15, 'R_WR': 16, 'L_EL': 13, 'R_EL': 14}
        for name, idx in key_points.items():
            coords[name] = np.array([lm[idx].x * width, lm[idx].y * height])
        return coords

    def detect_punch_v4(self, coords, dt):
        """ 
        V4 雙手監測與進度條：
        不僅判斷是否出拳，還判斷是哪隻手出拳，並回傳動作完成度(0~1.0)
        """
        if not self.prev_landmarks or dt <= 0:
            return 0.0, False, None, 0.0
            
        shoulder_dist_px = np.linalg.norm(coords['L_SH'] - coords['R_SH'])
        if shoulder_dist_px < 10: return 0.0, False, None, 0.0
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        hands_data = {}
        
        # 同時監測左右手
        for hand in ['LEFT', 'RIGHT']:
            sh_key = 'L_SH' if hand == 'LEFT' else 'R_SH'
            el_key = 'L_EL' if hand == 'LEFT' else 'R_EL'
            wr_key = 'L_WR' if hand == 'LEFT' else 'R_WR'
            
            # 手腕速度
            wrist_disp = np.linalg.norm(coords[wr_key] - self.prev_landmarks[wr_key])
            wrist_speed = (wrist_disp / pixels_per_meter) / dt
            
            # 手肘速度
            elbow_disp = np.linalg.norm(coords[el_key] - self.prev_landmarks[el_key])
            elbow_speed = (elbow_disp / pixels_per_meter) / dt
            
            # 手臂伸展狀態 (稍微放寬)
            curr_arm_len = np.linalg.norm(coords[sh_key] - coords[wr_key])
            prev_arm_len = np.linalg.norm(self.prev_landmarks[sh_key] - self.prev_landmarks[wr_key])
            is_extending = curr_arm_len > prev_arm_len + 2 
            
            # 綜合速度
            composite_speed = (wrist_speed * 0.6) + (elbow_speed * 0.4)
            hands_data[hand] = {'speed': composite_speed, 'is_extending': is_extending}
            
        # 找出當前移動最快的手
        fastest_hand = 'LEFT' if hands_data['LEFT']['speed'] > hands_data['RIGHT']['speed'] else 'RIGHT'
        max_speed = hands_data[fastest_hand]['speed']
        is_extending = hands_data[fastest_hand]['is_extending']
        
        # 觸發門檻 (超級敏感)
        threshold_normal = 0.5   # 正常出拳門檻
        threshold_explosive = 1.2 # 瞬間爆發門檻
        
        # 計算完成度 (0.0 ~ 1.0)
        completion = min(max_speed / threshold_normal, 1.0)
        
        # 觸發判定
        is_punch = False
        if (max_speed > threshold_normal and is_extending) or max_speed > threshold_explosive:
            is_punch = True
            
        return max_speed, is_punch, fastest_hand, completion

    def reset_game(self):
        self.state = 'WAIT_GUARD'
        self.current_round = 0
        self.hit_count = 0
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
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # ================= 狀態機 =================
        
        if self.state == 'GAME_OVER':
            # 1. 繪製結果背景
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.92, img, 0.08, 0)
            
            # 2. 結算數據
            l_rt = np.mean(self.left_stats['reaction']) if self.left_stats['reaction'] else 0
            l_sp = np.mean(self.left_stats['speed']) if self.left_stats['speed'] else 0
            r_rt = np.mean(self.right_stats['reaction']) if self.right_stats['reaction'] else 0
            r_sp = np.mean(self.right_stats['speed']) if self.right_stats['speed'] else 0
            total_avg_rt = (l_rt + r_rt) / 2 if (l_rt+r_rt) > 0 else 0
            
            # 命中率計算
            accuracy = (self.hit_count / 10) * 100
            
            # 評級
            rank = "C"
            rank_color = (200, 200, 200)
            if total_avg_rt > 0 and accuracy >= 50:
                if total_avg_rt < 250: rank, rank_color = "S (神速)", COLOR_CYAN
                elif total_avg_rt < 350: rank, rank_color = "A (優秀)", COLOR_GREEN
                elif total_avg_rt < 450: rank, rank_color = "B (普通)", COLOR_WARNING
            
            # === UI 繪製 ===
            cx = int(w/2)
            img = self.put_chinese_text(img, "=== 最終測驗報告 ===", (cx, int(h*0.08)), COLOR_TEXT, 50, 2, center_align=True)
            
            col_y_start = int(h * 0.22)
            line_gap = 55
            
            # 左手
            lx = int(w * 0.25)
            img = self.put_chinese_text(img, "【左手】", (lx, col_y_start), COLOR_CYAN, 45, 2, center_align=True)
            img = self.put_chinese_text(img, f"平均反應: {l_rt:.0f} ms", (lx, col_y_start + line_gap), COLOR_TEXT, 30, center_align=True)
            img = self.put_chinese_text(img, f"平均均速: {l_sp:.1f} m/s", (lx, col_y_start + line_gap*2), COLOR_TEXT, 30, center_align=True)
            
            # 右手
            rx = int(w * 0.75)
            img = self.put_chinese_text(img, "【右手】", (rx, col_y_start), COLOR_RED, 45, 2, center_align=True)
            img = self.put_chinese_text(img, f"平均反應: {r_rt:.0f} ms", (rx, col_y_start + line_gap), COLOR_TEXT, 30, center_align=True)
            img = self.put_chinese_text(img, f"平均均速: {r_sp:.1f} m/s", (rx, col_y_start + line_gap*2), COLOR_TEXT, 30, center_align=True)
            
            # 命中率與總評
            img = self.put_chinese_text(img, f"命中率: {accuracy:.0f}% ({self.hit_count}/10)", (cx, int(h*0.5)), COLOR_GREEN, 60, 3, center_align=True)
            img = self.put_chinese_text(img, f"綜合等級: {rank}", (cx, int(h*0.65)), rank_color, 70, 3, center_align=True)
            
            # 5 秒自動重啟倒數
            remain_time = 5.0 - (current_time - self.game_over_time)
            if remain_time <= 0:
                self.reset_game()
            else:
                img = self.put_chinese_text(img, f"系統將在 {int(remain_time)+1} 秒後重新開始...", (cx, h-80), (150,150,150), 30, center_align=True)

        elif self.state == 'WAIT_GUARD':
            hold_time = 3.0 if self.is_first_round else 2.0
            img = self.put_chinese_text(img, f"Round {self.current_round + 1}/10", (30, 50), COLOR_TEXT, 40, 2)
            
            elapsed = current_time - self.start_time
            remain = max(0.0, hold_time - elapsed)
            cx, cy = int(w/2), int(h/2)

            if coords:
                l_guard = coords['L_WR'][1] < coords['L_SH'][1] + 60
                r_guard = coords['R_WR'][1] < coords['R_SH'][1] + 60
                
                if l_guard and r_guard:
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
                 img = self.put_chinese_text(img, "偵測不到人像", (cx, cy), COLOR_ERROR, 40, 2, center_align=True)

        elif self.state == 'COUNTDOWN':
            delay = random.uniform(0.8, 2.0)
            if current_time - self.start_time > delay:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed_in_round = 0
            else:
                cv2.circle(img, (int(w/2), int(h/2)), 25, (255, 255, 255), -1)

        elif self.state == 'STIMULUS':
            text = "左拳!" if self.target == 'LEFT' else "右拳!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            img = self.put_chinese_text(img, text, (int(w/2), int(h/2)-50), color, 120, 5, center_align=True)
            
            if coords:
                speed, is_punch, punched_hand, completion = self.detect_punch_v4(coords, dt)
                
                # 繪製畫面下方的「動作完成度條」
                bar_w = int(w * 0.7)
                bar_h = 25
                start_x = int((w - bar_w) / 2)
                start_y = h - 60
                
                cv2.rectangle(img, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1) # 底色
                cv2.rectangle(img, (start_x, start_y), (start_x + int(bar_w * completion), start_y + bar_h), COLOR_WARNING, -1) # 進度
                cv2.rectangle(img, (start_x, start_y), (start_x + bar_w, start_y + bar_h), COLOR_TEXT, 2) # 邊框
                img = self.put_chinese_text(img, f"動作完成度: {int(completion*100)}%", (start_x, start_y - 35), COLOR_TEXT, 25)
                
                if speed > self.max_speed_in_round:
                    self.max_speed_in_round = speed
                
                if is_punch:
                    rt = (current_time - self.stimulus_time) * 1000
                    if rt > 60:
                        # 判定命中或失誤
                        is_hit = (punched_hand == self.target)
                        if is_hit:
                            self.hit_count += 1
                        
                        self.last_result = {
                            "reaction": rt,
                            "speed": self.max_speed_in_round,
                            "target": self.target,
                            "punched_hand": punched_hand,
                            "is_hit": is_hit
                        }
                        
                        # 依據實際出拳的手來記錄數據
                        if punched_hand == 'LEFT':
                            self.left_stats['reaction'].append(rt)
                            self.left_stats['speed'].append(self.max_speed_in_round)
                        else:
                            self.right_stats['reaction'].append(rt)
                            self.right_stats['speed'].append(self.max_speed_in_round)
                            
                        self.state = 'RESULT'
                        self.feedback_end_time = current_time + 1.2
                        self.current_round += 1
                        self.is_first_round = False
                        
                        # 如果是最後一局，記錄當前時間準備倒數
                        if self.current_round >= self.max_rounds:
                            self.game_over_time = current_time + 1.2

        elif self.state == 'RESULT':
            res = self.last_result
            cx, cy = int(w/2), int(h/2)
            
            # 顯示命中或失誤
            if res['is_hit']:
                img = self.put_chinese_text(img, "🎯 命中!", (cx, cy-100), COLOR_GREEN, 100, 4, center_align=True)
            else:
                # 顯示提示，告訴使用者他出錯手了
                wrong_txt = "出成右拳" if res['punched_hand'] == 'RIGHT' else "出成左拳"
                img = self.put_chinese_text(img, "❌ 失誤!", (cx, cy-120), COLOR_ERROR, 100, 4, center_align=True)
                img = self.put_chinese_text(img, f"({wrong_txt})", (cx, cy-40), COLOR_ERROR, 40, 2, center_align=True)
            
            # 顯示數據
            img = self.put_chinese_text(img, f"反應: {res['reaction']:.0f} ms", (cx, cy+40), COLOR_TEXT, 45, 2, center_align=True)
            img = self.put_chinese_text(img, f"速度: {res['speed']:.1f} m/s", (cx, cy+100), COLOR_TEXT, 45, 2, center_align=True)
            
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
    st.title("🥊 拳擊反應測試 V30 (命中判定版)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        webrtc_streamer(
            key="boxing-v30",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col2:
        st.markdown("### 🎯 V30 全新升級")
        st.markdown("""
        **1. 左右拳精準判定:**
        * 系統會判斷您是否出「對」手！
        * 提示左拳卻出右拳，將判定為 **失誤**。
        
        **2. 動作完成度進度條:**
        * 出拳瞬間，畫面下方會顯示您出拳的動能累積進度。
        
        **3. 全自動流程:**
        * 測驗結束後將顯示 **命中率**。
        * 顯示 5 秒後系統會 **自動重新開始**，無需手動點擊。
        """)

if __name__ == "__main__":
    main()
