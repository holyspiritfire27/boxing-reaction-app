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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.state = 'WAIT_GUARD' 
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        self.command_display_until = 0
        
        # 極簡化預備姿勢檢測
        self.guard_hold_start_time = None
        self.guard_stable_frames = 0
        self.guard_stable_threshold = 5  # 只需5幀
        self.guard_pose_valid = False
        
        # 防誤觸機制
        self.min_punch_duration = 0.15
        self.punch_start_time = None
        self.punch_detection_active = False
        self.false_trigger_count = 0
        self.false_trigger_threshold = 3
        
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
        
        # 歷史數據緩衝
        self.pos_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        self.prev_landmarks = None
        
        # === 物理參數 ===
        self.SHOULDER_WIDTH_M = 0.45 
        
        # 速度計算參數
        self.MIN_VELOCITY_THRESHOLD = 3.0
        self.MIN_ACCELERATION_THRESHOLD = 20.0
        self.ACC_WINDOW = 0.3
        self.Z_PUNCH_THRESHOLD = 0.12  # 降低深度門檻
        self.ARM_ANGLE_THRESHOLD = 120  # 降低角度門檻
        
        # 速度計算變數
        self.acc_start_time = None
        self.max_v_temp = 0.0
        self.max_acc_temp = 0.0
        self.prev_instant_v = 0.0
        self.filtered_v = 0.0
        
        # 速度平滑
        self.speed_smoothing_factor = 0.2
        self.smoothed_speed = 0.0
        
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
        """計算3D速度"""
        if dt <= 0 or prev_pos is None:
            return 0, 0
        
        # 計算前向速度（Z軸為主）
        dz = prev_pos.z - curr_pos.z  # 正值表示向前
        forward_velocity = max(0, dz * scale / dt)
        
        return forward_velocity, forward_velocity

    def calculate_speed_from_trajectory(self, positions, times, scale):
        """從軌跡計算速度和加速度"""
        if len(positions) < 3:
            return 0, 0, 0
        
        # 提取Z軸位置
        z_positions = [p.z * scale for p in positions]
        time_array = np.array(times) - times[0]
        
        if len(time_array) < 2:
            return 0, 0, 0
        
        # 計算速度
        velocities = []
        
        for i in range(1, len(z_positions)):
            if i < len(time_array):
                dt = time_array[i] - time_array[i-1]
                if dt > 0:
                    v = abs(z_positions[i-1] - z_positions[i]) / dt
                    velocities.append(v)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            peak_velocity = np.max(velocities)
            return avg_velocity, peak_velocity, peak_velocity * 2
        
        return 0, 0, 0

    def check_guard_pose(self, landmarks):
        """極簡化預備姿勢檢測 - 只需手部在頭部附近即可"""
        if landmarks is None:
            return False
        
        try:
            # 只檢查基本條件：手部在頭部附近
            nose = landmarks[0]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            
            # 檢查手腕是否在鼻子附近（Y軸）
            l_height_ok = abs(l_wrist.y - nose.y) < 0.3  # 寬鬆範圍
            r_height_ok = abs(r_wrist.y - nose.y) < 0.3
            
            # 檢查手腕是否在肩膀兩側（基本位置）
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            
            l_position_ok = l_wrist.x < l_shoulder.x + 0.2  # 左拳在左肩左側或附近
            r_position_ok = r_wrist.x > r_shoulder.x - 0.2  # 右拳在右肩右側或附近
            
            # 只要雙手都在頭部高度附近即可
            return (l_height_ok or r_height_ok)
            
        except:
            return False

    def get_speed_rating(self, speed):
        """速度評價"""
        if speed < 4.0: return "慢速"
        elif speed < 6.0: return "初學者"
        elif speed < 8.0: return "業餘"
        elif speed < 10.0: return "專業"
        elif speed < 13.0: return "選手級"
        elif speed < 16.0: return "世界級"
        else: return "傳奇"

    def get_reaction_rating(self, r_time):
        """反應時間評價"""
        if r_time > 300: return "遲緩"
        elif r_time > 200: return "一般"
        elif r_time >= 150: return "良好"
        elif r_time >= 120: return "優異"
        else: return "頂尖"

    def draw_status(self, image, h, w, state, target=None):
        """繪製狀態指示"""
        # 狀態文字
        status_texts = {
            'WAIT_GUARD': "請舉起雙手",
            'PRE_START': "準備開始...",
            'STIMULUS': "出拳！",
            'RESULT_PENDING': "計算中...",
            'RESULT': "完成"
        }
        
        status_text = status_texts.get(state, "準備中")
        
        # 繪製狀態框（右上角）
        box_width, box_height = 300, 80
        start_x, start_y = w - box_width - 20, 20
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + box_width, start_y + box_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + box_width, start_y + box_height), 
                     (0, 255, 255), 2)
        
        # 狀態文字
        cv2.putText(image, f"狀態: {status_text}", 
                   (start_x + 10, start_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 目標提示
        if target:
            target_text = f"目標: {target}"
            cv2.putText(image, target_text, 
                       (start_x + 10, start_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if target == 'LEFT' else (255, 0, 0), 2)
        
        return image

    def draw_prompt(self, image, h, w, target_side):
        """繪製出拳提示"""
        if target_side == 'LEFT':
            color = (0, 200, 255)  # 青色
            text = "左拳！"
        else:
            color = (255, 50, 150)  # 粉紅色
            text = "右拳！"
        
        # 位置（中央偏上）
        text_x = w // 2
        text_y = h // 3
        
        # 大文字
        font_scale = 4
        thickness = 10
        
        # 計算文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 背景
        padding = 40
        bg_x1 = text_x - text_width//2 - padding
        bg_y1 = text_y - text_height//2 - padding
        bg_x2 = text_x + text_width//2 + padding
        bg_y2 = text_y + text_height//2 + padding
        
        # 白色背景
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 8)
        
        # 繪製文字
        text_pos = (text_x - text_width//2, text_y + text_height//2)
        cv2.putText(image, text, text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 倒數計時
        if self.command_display_until > 0:
            remaining = max(0, self.command_display_until - time.time())
            countdown_text = f"{remaining:.1f}"
            (cw, ch), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            cv2.putText(image, countdown_text, 
                       (text_x - cw//2, text_y + 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        
        return image

    def draw_results(self, image, h, w):
        """繪製結果面板"""
        if not self.show_results:
            return image
            
        panel_height = 280
        start_y = h - panel_height
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, start_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 標題
        title = "出拳分析結果"
        cv2.putText(image, title, (20, start_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # 數據
        r_time_val = int(self.last_reaction_time)
        speed_val = self.last_punch_speed
        acc_val = self.last_punch_peak_acc
        
        r_rating = self.get_reaction_rating(r_time_val)
        s_rating = self.get_speed_rating(speed_val)
        
        y_offset = start_y + 80
        
        # 反應時間
        cv2.putText(image, f"反應時間: {r_time_val} ms", 
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
        cv2.putText(image, f"({r_rating})", 
                   (280, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 速度
        cv2.putText(image, f"出拳速度: {speed_val:.1f} m/s", 
                   (30, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
        cv2.putText(image, f"({s_rating})", 
                   (280, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        
        # 加速度
        cv2.putText(image, f"加速度: {acc_val:.0f} m/s²", 
                   (30, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
        
        # 分隔線
        cv2.line(image, (20, y_offset + 120), (w - 20, y_offset + 120), (100, 100, 100), 2)
        
        # 歷史次數
        total_tests = len(self.reaction_history)
        if total_tests > 0:
            avg_time = np.mean(self.reaction_history)
            avg_speed = np.mean(self.speed_history)
            
            cv2.putText(image, f"測試次數: {total_tests} 次", 
                       (30, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
            cv2.putText(image, f"平均反應: {int(avg_time)} ms | 平均速度: {avg_speed:.1f} m/s", 
                       (30, y_offset + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        
        return image

    def draw_speed_bar(self, image, h, w):
        """繪製速度條"""
        if self.state not in ['STIMULUS', 'RESULT_PENDING', 'RESULT']:
            return image
            
        bar_w, bar_h = 250, 20
        start_x, start_y = w - 270, h - 350
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        
        # 當前速度
        display_val = self.smoothed_speed if self.state == 'STIMULUS' else self.last_punch_speed
        
        # 比例
        display_ratio = min(1.0, display_val / 20.0)
        fill_w = int(display_ratio * bar_w)
        
        # 顏色
        if display_ratio < 0.3: 
            color = (0, 255, 255)
        elif display_ratio < 0.6:
            color = (0, 255, 0)
        elif display_ratio < 0.8:
            color = (0, 165, 255)
        else:
            color = (255, 0, 0)
        
        # 繪製速度條
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + fill_w, start_y + bar_h), color, -1)
        
        # 邊框
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + bar_w, start_y + bar_h), (200, 200, 200), 2)
        
        # 標籤
        cv2.putText(image, f"速度: {display_val:.1f} m/s", 
                   (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

    def draw_countdown(self, image, h, w, remaining_time):
        """繪製倒數計時"""
        if remaining_time <= 0:
            return image
            
        text = f"{remaining_time:.1f}"
        text_x = w // 2
        text_y = h // 2 + 100
        
        cv2.putText(image, text, (text_x - 50, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        
        return image

    def process(self, image):
        image.flags.writeable = False
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        h, w, _ = image.shape
        current_time = time.time()
        
        # 計算時間差
        dt = current_time - self.prev_time if self.prev_time > 0 else 0.033
        
        # 計算FPS
        if dt > 0:
            current_fps = 1.0 / dt
            self.current_fps = 0.9 * self.current_fps + 0.1 * current_fps
        
        # 繪製狀態
        image = self.draw_status(image, h, w, self.state, self.target)
        
        # 繪製FPS
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # 儲存歷史數據
            self.pos_history.append(landmarks)
            self.time_history.append(current_time)
            
            # === 簡化的狀態機邏輯 ===
            
            # 狀態 1: 等待預備姿勢
            if self.state == 'WAIT_GUARD':
                # 極簡檢測：只要手在頭部附近
                has_hands_up = False
                
                try:
                    nose = landmarks[0]
                    l_wrist = landmarks[15]
                    r_wrist = landmarks[16]
                    
                    # 檢查手腕是否在鼻子附近
                    l_ok = abs(l_wrist.y - nose.y) < 0.4
                    r_ok = abs(r_wrist.y - nose.y) < 0.4
                    has_hands_up = l_ok or r_ok
                except:
                    has_hands_up = False
                
                if has_hands_up:
                    self.guard_stable_frames += 1
                    
                    if self.guard_stable_frames >= self.guard_stable_threshold:
                        # 顯示準備訊息
                        cv2.putText(image, "準備就緒!", (w//2 - 100, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        # 短暫延遲後自動開始
                        if self.guard_hold_start_time is None:
                            self.guard_hold_start_time = current_time
                        
                        hold_duration = current_time - self.guard_hold_start_time
                        
                        if hold_duration > 0.5:  # 只需0.5秒
                            self.state = 'PRE_START'
                            self.wait_until = current_time + random.uniform(0.5, 1.5)  # 短隨機等待
                            self.guard_hold_start_time = None
                            self.guard_stable_frames = 0
                    else:
                        # 顯示進度
                        progress = int((self.guard_stable_frames / self.guard_stable_threshold) * 100)
                        cv2.putText(image, f"準備中... {progress}%", (w//2 - 100, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                else:
                    self.guard_stable_frames = 0
                    self.guard_hold_start_time = None
                    cv2.putText(image, "請舉起雙手", (w//2 - 100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            
            # 狀態 2: 預備開始
            elif self.state == 'PRE_START':
                # 顯示倒數
                remaining = self.wait_until - current_time
                if remaining > 0:
                    countdown_text = f"準備... {remaining:.1f}"
                    cv2.putText(image, countdown_text, (w//2 - 100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                else:
                    # 隨機選擇目標
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.2
                    
                    # 重置計數器
                    self.max_v_temp = 0.0
                    self.max_acc_temp = 0.0
                    self.acc_start_time = None
                    self.prev_instant_v = 0.0
                    self.filtered_v = 0.0
                    self.smoothed_speed = 0.0
                    self.punch_start_time = None
                    self.punch_detection_active = False
                    self.show_results = False
            
            # 狀態 3: 顯示提示
            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                if current_time <= self.command_display_until:
                    image = self.draw_prompt(image, h, w, self.target)
            
            # 狀態 4: 檢測出拳
            if self.state == 'STIMULUS':
                # 選擇目標手腕
                if self.target == 'LEFT':
                    wrist_idx, elbow_idx, shoulder_idx = 15, 13, 11
                else:
                    wrist_idx, elbow_idx, shoulder_idx = 16, 14, 12
                
                wrist = landmarks[wrist_idx]
                elbow = landmarks[elbow_idx]
                shoulder = landmarks[shoulder_idx]
                
                # 計算手臂角度
                angle = self.calculate_angle(shoulder, elbow, wrist) if elbow_idx in [13, 14] else 0
                
                # 計算速度
                velocity = 0
                
                if self.prev_landmarks and dt > 0:
                    prev_wrist = self.prev_landmarks[wrist_idx]
                    
                    # 計算比例尺
                    l_sh = landmarks[11]
                    r_sh = landmarks[12]
                    sh_dist_2d = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
                    scale = self.SHOULDER_WIDTH_M / sh_dist_2d if sh_dist_2d > 0 else 1.0
                    
                    # 計算速度
                    forward_v, _ = self.calculate_3d_velocity(wrist, prev_wrist, scale, dt)
                    velocity = forward_v
                    
                    # 平滑處理
                    self.smoothed_speed = (self.smoothed_speed * 0.7 + velocity * 0.3)
                    
                    # 計算加速度
                    acceleration = 0
                    if self.prev_instant_v > 0 and dt > 0:
                        acceleration = (velocity - self.prev_instant_v) / dt
                    
                    self.prev_instant_v = velocity
                
                # 檢測出拳開始
                if velocity > self.MIN_VELOCITY_THRESHOLD:
                    if self.punch_start_time is None:
                        self.punch_start_time = current_time
                        self.punch_detection_active = True
                
                # 更新最大值
                if self.punch_detection_active:
                    self.max_v_temp = max(self.max_v_temp, self.smoothed_speed)
                    self.max_acc_temp = max(self.max_acc_temp, acceleration if 'acceleration' in locals() else 0)
                
                # 擊中條件（簡化）
                cond_duration = (self.punch_start_time is not None and 
                                (current_time - self.punch_start_time) > self.min_punch_duration)
                cond_speed = self.max_v_temp > self.MIN_VELOCITY_THRESHOLD
                cond_angle = angle > self.ARM_ANGLE_THRESHOLD if angle > 0 else True
                cond_forward = (shoulder.z - wrist.z) > self.Z_PUNCH_THRESHOLD
                
                # 判定擊中
                if cond_duration and cond_speed and (cond_angle or cond_forward):
                    
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    self.last_punch_speed = min(25.0, self.max_v_temp)
                    self.last_punch_peak_acc = self.max_acc_temp
                    
                    # 保存數據
                    self.reaction_history.append(self.last_reaction_time)
                    self.speed_history.append(self.last_punch_speed)
                    self.acc_history.append(self.last_punch_peak_acc)
                    
                    self.show_results = True
                    self.state = 'RESULT_PENDING'
                    self.wait_until = current_time + 1.0
                
                # 超時處理
                if (current_time - self.start_time) > 3.0:
                    self.state = 'WAIT_GUARD'
            
            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0
            
            elif self.state == 'RESULT':
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'
            
            self.prev_landmarks = landmarks
        
        else:
            # 沒有檢測到姿勢
            if self.state == 'WAIT_GUARD':
                cv2.putText(image, "請面對鏡頭", (w//2 - 100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        
        self.prev_time = current_time
        
        # 繪製結果和速度條
        if self.show_results:
            image = self.draw_results(image, h, w)
        
        image = self.draw_speed_bar(image, h, w)
        
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
    st.set_page_config(page_title="拳擊反應測試", layout="wide")
    
    st.title("🥊 簡易拳擊反應測試")
    
    with st.sidebar:
        st.header("快速開始")
        
        st.markdown("### 只需三步：")
        st.markdown("1. **面對鏡頭**")
        st.markdown("2. **舉起雙手**（任意姿勢）")
        st.markdown("3. **看到提示後出拳**")
        
        st.divider()
        
        st.info("💡 **提示**")
        st.markdown("- 系統會自動檢測你的姿勢")
        st.markdown("- 看到『左拳！』或『右拳！』就快速出拳")
        st.markdown("- 不需要特定預備姿勢")
        
        st.divider()
        
        if st.button("🔄 重新開始測試"):
            st.runtime.legacy_caching.clear_cache()
            st.experimental_rerun()
    
    # 主畫面
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("即時分析")
        
        # 創建容器
        video_container = st.empty()
        
        # 顯示使用提示
        with video_container.container():
            st.info("👆 點擊下方按鈕啟動攝影機")
        
        # 啟動按鈕
        if st.button("🎥 啟動攝影機", type="primary"):
            video_container.empty()
            
            # 寬容的媒體約束
            media_stream_constraints = {
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            }
            
            ctx = webrtc_streamer(
                key="simple-boxing-test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints=media_stream_constraints,
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )
            
            if not ctx.state.playing:
                st.warning("請允許攝影機權限")
    
    with col2:
        st.subheader("評分標準")
        
        st.markdown("**速度等級：**")
        st.markdown("- < 4 m/s: 慢速")
        st.markdown("- 4-6 m/s: 初學者")
        st.markdown("- 6-8 m/s: 業餘")
        st.markdown("- 8-10 m/s: 專業")
        st.markdown("- 10-13 m/s: 選手級")
        st.markdown("- > 13 m/s: 世界級")
        
        st.markdown("**反應時間：**")
        st.markdown("- < 120 ms: 頂尖")
        st.markdown("- 120-150 ms: 優異")
        st.markdown("- 150-200 ms: 良好")
        st.markdown("- 200-300 ms: 一般")
        st.markdown("- > 300 ms: 遲緩")
        
        st.divider()
        
        st.markdown("**世界紀錄參考：**")
        st.markdown("- 職業拳手: 8-12 m/s")
        st.markdown("- 頂尖選手: 12-15 m/s")
        st.markdown("- 最快反應: 100-120 ms")


if __name__ == "__main__":
    main()
