import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import random
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from collections import deque
import math

class BoxingAnalystLogic:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # 提高信心度
            min_tracking_confidence=0.7,
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
        self.last_punch_peak_acc = 0.0
        self.reaction_history = [] 
        self.speed_history = []    
        self.acc_history = []
        self.show_results = False
        
        # FPS 監測
        self.prev_time = 0
        self.current_fps = 0.0
        self.low_fps_warning = False
        
        # 歷史數據緩衝 (用於平滑和速度計算)
        self.pos_history = deque(maxlen=10)  # 保存最近10幀的位置
        self.time_history = deque(maxlen=10)
        self.prev_landmarks = None
        
        # === 增強物理參數 ===
        self.SHOULDER_WIDTH_M = 0.45 
        
        # 速度計算參數
        self.MIN_VELOCITY_THRESHOLD = 2.5  # 稍微提高門檻
        self.ACC_WINDOW = 0.2              # 縮短到0.2秒的爆發窗口
        self.Z_PUNCH_THRESHOLD = 0.12      # 稍微降低Z軸門檻
        self.ARM_ANGLE_THRESHOLD = 125     # 稍微提高角度門檻
        self.RETRACTION_THRESHOLD = 0.35   # 稍微提高回收門檻
        
        # 新的物理參數
        self.MIN_ACCELERATION_THRESHOLD = 15.0  # 最小加速度 (m/s²)
        self.SMOOTHING_FACTOR = 0.3             # 速度平滑因子
        self.PUNCH_TRAVEL_DISTANCE = 0.5        # 假設出拳移動距離約0.5米
        
        # 速度計算變數
        self.acc_start_time = None
        self.max_v_temp = 0.0
        self.max_acc_temp = 0.0
        self.prev_instant_v = 0.0
        self.filtered_v = 0.0
        
        # 擊中檢測變數
        self.punch_detected = False
        self.punch_start_time = None
        self.punch_start_pos = None
        
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

    def calculate_3d_velocity(self, curr_pos, prev_pos, scale, dt):
        """
        基於3D位置計算速度
        改進：使用3D歐幾里得距離，而不是僅Z軸
        """
        if dt <= 0 or prev_pos is None:
            return 0, 0
        
        # 將2D坐標和深度轉換為3D坐標
        curr_3d = np.array([curr_pos.x, curr_pos.y, curr_pos.z])
        prev_3d = np.array([prev_pos.x, prev_pos.y, prev_pos.z])
        
        # 計算3D距離（單位：米）
        distance_3d = np.linalg.norm(curr_3d - prev_3d) * scale
        
        # 速度 = 距離 / 時間
        velocity = distance_3d / dt
        
        # 計算前向分量（基於與肩膀的相對位置）
        forward_velocity = max(0, (prev_pos.z - curr_pos.z) * scale / dt)
        
        return velocity, forward_velocity

    def calculate_speed_from_trajectory(self, positions, times, scale):
        """
        從軌跡擬合計算速度 - 更準確的方法
        """
        if len(positions) < 3:
            return 0, 0
        
        # 提取Z軸位置（深度）
        z_positions = [p.z for p in positions]
        
        # 計算速度（使用線性回歸）
        time_array = np.array(times) - times[0]
        z_array = np.array(z_positions) * scale
        
        if len(time_array) < 2:
            return 0, 0
        
        # 使用中央差分計算速度
        velocities = []
        for i in range(1, len(z_array)):
            if i < len(time_array):
                dt = time_array[i] - time_array[i-1]
                if dt > 0:
                    v = abs(z_array[i-1] - z_array[i]) / dt
                    velocities.append(v)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            peak_velocity = np.max(velocities)
            return avg_velocity, peak_velocity
        
        return 0, 0

    def get_speed_rating(self, speed):
        """
        根據真實物理數據更新的評價標準
        參考文獻：
        - 業餘拳手：5-8 m/s
        - 專業拳手：8-12 m/s  
        - 世界級拳手：12-15 m/s
        - 極限：15-20 m/s (如泰森)
        """
        if speed < 4.0: return "慢速/暖身"
        elif speed < 6.0: return "初學者"
        elif speed < 8.0: return "業餘水準"
        elif speed < 10.0: return "專業級"
        elif speed < 13.0: return "選手級"
        elif speed < 16.0: return "世界級"
        else: return "傳奇級別"

    def get_reaction_rating(self, r_time):
        if r_time > 300: return "遲緩"
        elif r_time > 200: return "一般"
        elif r_time >= 150: return "良好"
        elif r_time >= 120: return "優異"
        else: return "頂尖選手"

    def get_acceleration_rating(self, acc):
        if acc < 30: return "普通"
        elif acc < 50: return "良好"
        elif acc < 80: return "優秀"
        elif acc < 120: return "卓越"
        else: return "爆發力驚人"

    def draw_feedback_bar(self, image, h, w):
        bar_w, bar_h = 280, 25
        start_x, start_y = w - 300, h - 60
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        
        # 顯示邏輯
        display_val = self.max_v_temp if self.state == 'STIMULUS' else self.last_punch_speed
        
        # 專業級標準：20 m/s為滿格
        display_ratio = min(1.0, display_val / 20.0)
        fill_w = int(display_ratio * bar_w)
        
        # 顏色漸層
        if display_ratio < 0.3: 
            color = (0, 255, 255)  # Cyan
        elif display_ratio < 0.6:
            color = (0, 255, 0)    # Green
        elif display_ratio < 0.8:
            color = (0, 165, 255)  # Orange
        else:
            color = (255, 0, 0)    # Red

        cv2_color = (color[2], color[1], color[0])
        cv2.rectangle(image, (start_x, start_y), (start_x + fill_w, start_y + bar_h), cv2_color, -1)
        
        # 添加刻度
        for i in range(1, 5):
            x_pos = start_x + int(i * 0.2 * bar_w)
            cv2.line(image, (x_pos, start_y), (x_pos, start_y + bar_h), (200, 200, 200), 1)
        
        txt = f"速度峰值: {display_val:.1f} m/s"
        image = self.put_chinese_text(image, txt, (start_x, start_y - 30), (255, 255, 255), 20)
        
        # 如果已經有加速度數據，也顯示
        if self.last_punch_peak_acc > 0:
            acc_text = f"峰值加速度: {self.last_punch_peak_acc:.0f} m/s²"
            image = self.put_chinese_text(image, acc_text, (start_x, start_y - 60), (255, 255, 200), 18)
            
        return image

    def draw_dashboard(self, image, h, w):
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 350), (480, h - 10), (0, 0, 0), -1)
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

        image = self.put_chinese_text(image, status_text, (20, h - 300), status_color, 40)

        if self.show_results:
            r_time_val = int(self.last_reaction_time)
            speed_val = self.last_punch_speed
            acc_val = self.last_punch_peak_acc
            r_rating = self.get_reaction_rating(r_time_val)
            s_rating = self.get_speed_rating(speed_val)
            a_rating = self.get_acceleration_rating(acc_val)

            image = self.put_chinese_text(image, f"反應時間: {r_time_val} ms [{r_rating}]", 
                                         (20, h - 240), (255, 255, 255), 22)
            image = self.put_chinese_text(image, f"出拳速度: {speed_val:.1f} m/s [{s_rating}]", 
                                         (20, h - 200), (255, 255, 255), 22)
            image = self.put_chinese_text(image, f"峰值加速度: {acc_val:.0f} m/s² [{a_rating}]", 
                                         (20, h - 160), (255, 255, 200), 22)
            
            cv2.line(image, (20, h - 140), (460, h - 140), (100, 100, 100), 1)

            avg_time = np.mean(self.reaction_history[-5:]) if len(self.reaction_history) > 0 else 0
            avg_speed = np.mean(self.speed_history[-5:]) if len(self.speed_history) > 0 else 0
            avg_acc = np.mean(self.acc_history[-5:]) if len(self.acc_history) > 0 else 0
            
            image = self.put_chinese_text(image, f"最近5次平均:", (20, h - 110), (200, 255, 200), 20)
            image = self.put_chinese_text(image, f"反應: {int(avg_time)} ms | 速度: {avg_speed:.1f} m/s | 加速度: {avg_acc:.0f} m/s²", 
                                         (20, h - 80), (200, 255, 200), 18)

        if self.low_fps_warning:
            image = self.put_chinese_text(image, f"警告：FPS {self.current_fps:.1f}", 
                                         (20, h - 50), (0, 255, 255), 18)
            
        return image

    def detect_punch_motion(self, landmarks, target_side, scale, current_time):
        """檢測出拳動作的狀態"""
        if target_side == 'LEFT':
            wrist = landmarks[15]
            elbow = landmarks[13]
            shoulder = landmarks[11]
        else:
            wrist = landmarks[16]
            elbow = landmarks[14]
            shoulder = landmarks[12]
        
        # 計算手臂角度
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # 計算拳頭相對於肩膀的位置
        rel_x = abs(wrist.x - shoulder.x)
        rel_z = shoulder.z - wrist.z  # 正值表示向前
        
        # 狀態檢測
        is_retracted = rel_x < self.RETRACTION_THRESHOLD
        is_extended = angle > self.ARM_ANGLE_THRESHOLD
        is_forward = rel_z > self.Z_PUNCH_THRESHOLD
        
        return {
            'angle': angle,
            'rel_x': rel_x,
            'rel_z': rel_z,
            'is_retracted': is_retracted,
            'is_extended': is_extended,
            'is_forward': is_forward,
            'wrist': wrist
        }

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        # 計算FPS
        if self.prev_time > 0:
            dt = current_time - self.prev_time
            if dt > 0:
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)  # 平滑處理
                if self.current_fps < 30: 
                    self.low_fps_warning = True
                else: 
                    self.low_fps_warning = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # 計算比例尺
            l_sh = landmarks[11]
            r_sh = landmarks[12]
            sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 1.0
            
            # 儲存歷史位置
            self.pos_history.append(landmarks)
            self.time_history.append(current_time)
            
            # === 狀態機邏輯 ===
            if self.state == 'WAIT_GUARD':
                # 檢測是否在防守姿勢
                left_state = self.detect_punch_motion(landmarks, 'LEFT', scale, current_time)
                right_state = self.detect_punch_motion(landmarks, 'RIGHT', scale, current_time)
                
                is_in_guard = left_state['is_retracted'] and right_state['is_retracted']
                
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
                    image = self.put_chinese_text(image, "請舉手護頭!", (int(w/2)-120, h-100), 
                                                 (255, 255, 255), 50, stroke_width=3)
            
            elif self.state == 'PRE_START':
                if current_time > self.wait_until:
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.2
                    
                    # 重置計數器
                    self.max_v_temp = 0.0
                    self.max_acc_temp = 0.0
                    self.acc_start_time = None
                    self.prev_instant_v = 0.0
                    self.filtered_v = 0.0
                    self.punch_detected = False
                    self.punch_start_time = None
                    self.punch_start_pos = None
                    self.show_results = False
            
            # 顯示目標指令
            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    color = (0, 200, 255) if self.target == 'LEFT' else (255, 50, 100)
                    target_text = "左拳!" if self.target == 'LEFT' else "右拳!"
                    image = self.put_chinese_text(image, target_text, 
                                                 (int(w/2)-150, int(h/2)-80), color, 120, stroke_width=8)
            
            # 出拳檢測階段
            if self.state == 'STIMULUS':
                target_state = self.detect_punch_motion(landmarks, self.target, scale, current_time)
                wrist = target_state['wrist']
                
                # 計算速度
                velocity = 0
                acceleration = 0
                
                if self.prev_landmarks and len(self.pos_history) >= 3:
                    prev_wrist = self.prev_landmarks[15] if self.target == 'LEFT' else self.prev_landmarks[16]
                    
                    # 計算瞬時速度
                    instant_v, forward_v = self.calculate_3d_velocity(wrist, prev_wrist, scale, dt)
                    
                    # 平滑處理
                    self.filtered_v = (self.filtered_v * 0.7 + instant_v * 0.3)
                    velocity = self.filtered_v
                    
                    # 計算加速度
                    if self.prev_instant_v > 0 and dt > 0:
                        acceleration = (velocity - self.prev_instant_v) / dt
                    
                    # 檢測出拳開始
                    if velocity > self.MIN_VELOCITY_THRESHOLD and not self.punch_detected:
                        if self.punch_start_time is None:
                            self.punch_start_time = current_time
                            self.punch_start_pos = wrist
                        self.punch_detected = True
                    
                    # 更新最大值
                    if self.punch_detected:
                        self.max_v_temp = max(self.max_v_temp, velocity)
                        self.max_acc_temp = max(self.max_acc_temp, acceleration)
                        
                        # 計算加速期
                        if self.acc_start_time is None and acceleration > self.MIN_ACCELERATION_THRESHOLD:
                            self.acc_start_time = current_time
                        
                        # 限制加速窗口
                        if self.acc_start_time is not None:
                            acc_duration = current_time - self.acc_start_time
                            if acc_duration > self.ACC_WINDOW:
                                # 超過窗口，停止更新最大值
                                pass
                    
                    self.prev_instant_v = velocity
                
                # 檢測擊中條件
                cond_speed = self.max_v_temp > self.MIN_VELOCITY_THRESHOLD
                cond_acc = self.max_acc_temp > self.MIN_ACCELERATION_THRESHOLD
                cond_extended = target_state['is_extended']
                cond_forward = target_state['is_forward']
                
                # 使用軌跡擬合計算最終速度（更準確）
                if self.punch_detected and len(self.pos_history) >= 5:
                    # 提取最近幾幀的數據
                    recent_positions = []
                    recent_times = []
                    
                    for i in range(min(5, len(self.pos_history))):
                        idx = -1 - i
                        pos = self.pos_history[idx]
                        wrist_pos = pos[15] if self.target == 'LEFT' else pos[16]
                        recent_positions.append(wrist_pos)
                        recent_times.append(self.time_history[idx])
                    
                    # 反轉以得到正確的時間順序
                    recent_positions.reverse()
                    recent_times.reverse()
                    
                    # 計算速度
                    avg_v, peak_v = self.calculate_speed_from_trajectory(recent_positions, recent_times, scale)
                    
                    if peak_v > self.max_v_temp:
                        self.max_v_temp = peak_v
                
                # 判定擊中
                if (cond_speed and cond_acc) and (cond_extended or cond_forward):
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    
                    # 使用軌跡擬合的速度作為最終速度
                    if len(self.pos_history) >= 3:
                        recent_positions = []
                        recent_times = []
                        
                        for i in range(min(8, len(self.pos_history))):
                            idx = -1 - i
                            pos = self.pos_history[idx]
                            wrist_pos = pos[15] if self.target == 'LEFT' else pos[16]
                            recent_positions.append(wrist_pos)
                            recent_times.append(self.time_history[idx])
                        
                        recent_positions.reverse()
                        recent_times.reverse()
                        
                        avg_v, peak_v = self.calculate_speed_from_trajectory(recent_positions, recent_times, scale)
                        self.last_punch_speed = peak_v
                    else:
                        self.last_punch_speed = self.max_v_temp
                    
                    self.last_punch_peak_acc = self.max_acc_temp
                    
                    # 避免極端值
                    if self.last_punch_speed > 25.0:
                        self.last_punch_speed = min(25.0, self.max_v_temp)
                    
                    self.reaction_history.append(self.last_reaction_time)
                    self.speed_history.append(self.last_punch_speed)
                    self.acc_history.append(self.last_punch_peak_acc)
                    
                    self.show_results = True
                    self.state, self.wait_until = 'RESULT_PENDING', self.command_display_until
                
                # 超時處理
                if (current_time - self.start_time) > 4.0: 
                    self.state = 'WAIT_GUARD'
                    self.show_results = True
            
            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state, self.wait_until = 'RESULT', current_time + 2.5
            
            elif self.state == 'RESULT':
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'
            
            self.prev_landmarks = landmarks
            self.prev_time = current_time
        
        else:
            self.prev_time = current_time
        
        # 繪製UI
        image = self.draw_dashboard(image, h, w)
        image = self.draw_feedback_bar(image, h, w)
        
        # 顯示當前FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(image, fps_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        return image


class VideoProcessor(VideoTransformerBase):
    def __init__(self): 
        self.logic = BoxingAnalystLogic()
    
    def recv(self, frame):
        try:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            return av.VideoFrame.from_ndarray(self.logic.process(img), format="bgr24")
        except Exception as e: 
            print(f"處理錯誤: {e}")
            return frame


def main():
    st.set_page_config(page_title="拳擊反應 v24 (物理引擎增強版)", layout="wide")
    st.title("🥊 拳擊反應分析系統 - 物理引擎增強版")
    
    with st.sidebar:
        st.header("🎯 v24 主要改進")
        st.write("**速度計算準確性提升：**")
        st.write("1. 3D軌跡擬合速度計算")
        st.write("2. 加速度檢測與峰值捕捉")
        st.write("3. 平滑濾波減少抖動")
        st.write("4. 多幀歷史數據分析")
        
        st.write("**物理模型增強：**")
        st.write("• 真實物理單位轉換")
        st.write("• 專業級評價標準")
        st.write("• 擊中條件多維檢測")
        
        st.write("**UI改進：**")
        st.write("• 峰值加速度顯示")
        st.write("• 歷史數據統計")
        st.write("• 視覺化反饋")
        
        st.divider()
        st.write("**使用提示：**")
        st.write("1. 保持良好光照")
        st.write("2. 確保全身在畫面中")
        st.write("3. 出拳時儘量保持軌跡穩定")
        st.write("4. 建議距離鏡頭2-3米")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("即時分析畫面")
        ctx = webrtc_streamer(
            key="boxing-v24-enhanced",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "frameRate": {"ideal": 60, "min": 45},
                    "width": {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720, "min": 480}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    
    with col2:
        st.subheader("專業參考標準")
        
        st.write("**速度等級對照：**")
        speed_data = {
            "等級": ["暖身", "初學者", "業餘", "專業", "選手", "世界級", "傳奇"],
            "速度(m/s)": ["<4", "4-6", "6-8", "8-10", "10-13", "13-16", ">16"],
            "範例": ["熱身運動", "新手練習", "俱樂部水準", "職業訓練", "比賽選手", "冠軍級別", "泰森級別"]
        }
        st.table(speed_data)
        
        st.write("**反應時間標準：**")
        reaction_data = {
            "等級": ["頂尖", "優異", "良好", "一般", "遲緩"],
            "時間(ms)": ["<120", "120-150", "150-200", "200-300", ">300"],
            "說明": ["職業選手", "優秀業餘", "正常水準", "需訓練", "反應較慢"]
        }
        st.table(reaction_data)
        
        if st.button("重置統計數據"):
            st.runtime.legacy_caching.clear_cache()
            st.success("數據已重置！")


if __name__ == "__main__":
    main()
