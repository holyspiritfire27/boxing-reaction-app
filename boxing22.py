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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        self.state = 'WAIT_GUARD' 
        self.target = None
        self.start_time = 0
        self.wait_until = 0
        self.command_display_until = 0
        
        # 改進：更嚴格的預備姿勢檢測
        self.guard_hold_start_time = None
        self.guard_stable_frames = 0
        self.guard_stable_threshold = 15  # 需要連續15幀穩定
        self.guard_pose_valid = False
        
        # 改進：防誤觸機制
        self.min_punch_duration = 0.15  # 最短出拳持續時間
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
        self.pos_history = deque(maxlen=15)
        self.time_history = deque(maxlen=15)
        self.prev_landmarks = None
        
        # === 改進的物理參數 ===
        self.SHOULDER_WIDTH_M = 0.45 
        
        # 更嚴格的門檻防止誤觸
        self.MIN_VELOCITY_THRESHOLD = 3.5  # 提高門檻
        self.MIN_ACCELERATION_THRESHOLD = 25.0  # 提高加速度門檻
        self.ACC_WINDOW = 0.3  # 稍微延長加速窗口
        self.Z_PUNCH_THRESHOLD = 0.18  # 提高深度門檻
        self.ARM_ANGLE_THRESHOLD = 130  # 提高角度門檻
        
        # 預備姿勢參數
        self.GUARD_ANGLE_MIN = 80  # 手肘最小角度
        self.GUARD_ANGLE_MAX = 120  # 手肘最大角度
        self.GUARD_HEIGHT_RATIO = 0.85  # 拳頭相對於頭部的高度
        
        # 速度計算變數
        self.acc_start_time = None
        self.max_v_temp = 0.0
        self.max_acc_temp = 0.0
        self.prev_instant_v = 0.0
        self.filtered_v = 0.0
        
        # 改進：速度平滑
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
        
        # 計算總速度（考慮3D移動）
        dx = (prev_pos.x - curr_pos.x) * scale
        dy = (prev_pos.y - curr_pos.y) * scale
        total_velocity = np.sqrt(dx**2 + dy**2 + (dz**2)) / dt
        
        return total_velocity, forward_velocity

    def calculate_speed_from_trajectory(self, positions, times, scale):
        """從軌跡計算速度和加速度"""
        if len(positions) < 4:
            return 0, 0, 0
        
        # 提取Z軸位置（深度）
        z_positions = [p.z * scale for p in positions]
        time_array = np.array(times) - times[0]
        
        if len(time_array) < 2:
            return 0, 0, 0
        
        # 計算速度和加速度
        velocities = []
        accelerations = []
        
        for i in range(1, len(z_positions)):
            if i < len(time_array):
                dt = time_array[i] - time_array[i-1]
                if dt > 0:
                    v = abs(z_positions[i-1] - z_positions[i]) / dt
                    velocities.append(v)
                    
                    # 計算加速度
                    if i > 1 and (time_array[i-1] - time_array[i-2]) > 0:
                        prev_v = abs(z_positions[i-2] - z_positions[i-1]) / (time_array[i-1] - time_array[i-2])
                        a = (v - prev_v) / dt if dt > 0 else 0
                        accelerations.append(a)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            peak_velocity = np.max(velocities)
            peak_acceleration = np.max(accelerations) if accelerations else 0
            return avg_velocity, peak_velocity, peak_acceleration
        
        return 0, 0, 0

    def check_guard_pose(self, landmarks):
        """檢查是否處於正確的預備姿勢"""
        if landmarks is None:
            return False
        
        try:
            # 關鍵點
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_elbow = landmarks[13]
            r_elbow = landmarks[14]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            nose = landmarks[0]
            
            # 計算手臂角度
            l_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # 檢查角度是否在合理範圍
            angle_ok = (self.GUARD_ANGLE_MIN < l_angle < self.GUARD_ANGLE_MAX and 
                       self.GUARD_ANGLE_MIN < r_angle < self.GUARD_ANGLE_MAX)
            
            # 檢查拳頭高度（應該在頭部附近）
            l_height_ok = abs(l_wrist.y - nose.y) < 0.15
            r_height_ok = abs(r_wrist.y - nose.y) < 0.15
            
            # 檢查拳頭與肩膀的相對位置（應該在頭部兩側）
            l_position_ok = abs(l_wrist.x - l_shoulder.x) < 0.2
            r_position_ok = abs(r_wrist.x - r_shoulder.x) < 0.2
            
            # 檢查對稱性
            symmetry_ok = abs(l_angle - r_angle) < 20
            
            return (angle_ok and l_height_ok and r_height_ok and 
                   l_position_ok and r_position_ok and symmetry_ok)
            
        except Exception as e:
            return False

    def get_speed_rating(self, speed):
        """速度評價"""
        if speed < 4.0: return "慢速/暖身"
        elif speed < 6.0: return "初學者"
        elif speed < 8.0: return "業餘水準"
        elif speed < 10.0: return "專業級"
        elif speed < 13.0: return "選手級"
        elif speed < 16.0: return "世界級"
        else: return "傳奇級別"

    def get_reaction_rating(self, r_time):
        """反應時間評價"""
        if r_time > 300: return "遲緩"
        elif r_time > 200: return "一般"
        elif r_time >= 150: return "良好"
        elif r_time >= 120: return "優異"
        else: return "頂尖選手"

    def draw_guard_indicator(self, image, h, w, guard_valid, progress):
        """繪製預備姿勢指示器"""
        # 位置調整到左上角，不擋住提示
        start_x, start_y = 20, 20
        box_width, box_height = 400, 100
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + box_width, start_y + box_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        if guard_valid:
            status_text = "✓ 預備姿勢正確"
            status_color = (0, 255, 0)
            instruction = f"保持姿勢... {progress}%"
        else:
            status_text = "請舉手做好預備姿勢"
            status_color = (0, 165, 255)
            instruction = "雙手舉起，拳頭在臉頰兩側"
        
        image = self.put_chinese_text(image, status_text, 
                                     (start_x + 10, start_y + 30), status_color, 28)
        image = self.put_chinese_text(image, instruction, 
                                     (start_x + 10, start_y + 70), (255, 255, 255), 22)
        
        return image

    def draw_prompt(self, image, h, w, target_side):
        """繪製出拳提示（最前方顯示）"""
        # 使用鮮明的顏色和大字體
        if target_side == 'LEFT':
            color = (0, 200, 255)  # 青色
            text = "左 拳 !"
        else:
            color = (255, 50, 150)  # 粉紅色
            text = "右 拳 !"
        
        # 計算文字位置（屏幕中央偏上）
        text_x = w // 2
        text_y = h // 3
        
        # 繪製文字背景（不透明白色）
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
        
        bg_x1 = text_x - text_width//2 - 30
        bg_y1 = text_y - text_height//2 - 20
        bg_x2 = text_x + text_width//2 + 30
        bg_y2 = text_y + text_height//2 + 20
        
        # 白色背景
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 8)
        
        # 繪製文字
        if self.use_chinese:
            # 使用中文字體
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_path, 100)
            
            # 計算文字位置
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((text_x - text_width//2, text_y - text_height//2), 
                     text, font=font, fill=color, stroke_width=6, stroke_fill=(0, 0, 0))
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # 使用英文字體
            cv2.putText(image, text, (text_x - text_width//2, text_y + text_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)
        
        # 添加倒數計時
        if self.command_display_until > 0:
            remaining = max(0, self.command_display_until - time.time())
            if remaining < 1.0:
                countdown = f"{remaining:.1f}"
                cv2.putText(image, countdown, (text_x - 50, text_y + 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        
        return image

    def draw_results(self, image, h, w):
        """繪製結果面板（底部）"""
        panel_height = 340
        start_y = h - panel_height
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, start_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 結果標題
        title_y = start_y + 50
        title = "🏆 本次出拳數據"
        
        if self.use_chinese:
            image = self.put_chinese_text(image, title, (20, title_y), (255, 255, 0), 32)
        else:
            cv2.putText(image, title, (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # 主要數據
        r_time_val = int(self.last_reaction_time)
        speed_val = self.last_punch_speed
        acc_val = self.last_punch_peak_acc
        
        r_rating = self.get_reaction_rating(r_time_val)
        s_rating = self.get_speed_rating(speed_val)
        
        # 數據顯示
        data_start_y = title_y + 60
        
        if self.use_chinese:
            image = self.put_chinese_text(image, f"⚡ 反應時間: {r_time_val} ms", 
                                         (30, data_start_y), (200, 255, 200), 26)
            image = self.put_chinese_text(image, f"[{r_rating}]", 
                                         (300, data_start_y), (255, 255, 0), 24)
            
            image = self.put_chinese_text(image, f"💨 出拳速度: {speed_val:.1f} m/s", 
                                         (30, data_start_y + 50), (200, 255, 200), 26)
            image = self.put_chinese_text(image, f"[{s_rating}]", 
                                         (300, data_start_y + 50), (255, 200, 0), 24)
            
            image = self.put_chinese_text(image, f"🚀 峰值加速度: {acc_val:.0f} m/s²", 
                                         (30, data_start_y + 100), (200, 255, 200), 26)
        else:
            cv2.putText(image, f"Reaction: {r_time_val} ms [{r_rating}]", 
                       (30, data_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
            cv2.putText(image, f"Speed: {speed_val:.1f} m/s [{s_rating}]", 
                       (30, data_start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
            cv2.putText(image, f"Acceleration: {acc_val:.0f} m/s²", 
                       (30, data_start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
        
        return image

    def draw_speed_bar(self, image, h, w):
        """繪製速度條"""
        bar_w, bar_h = 300, 25
        start_x, start_y = w - 320, 80  # 移到右上方
        
        # 背景
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + bar_w, start_y + bar_h), (50, 50, 50), -1)
        
        # 當前速度值
        if self.state == 'STIMULUS':
            display_val = self.smoothed_speed
        else:
            display_val = self.last_punch_speed
        
        # 比例計算（20 m/s為滿格）
        display_ratio = min(1.0, display_val / 20.0)
        fill_w = int(display_ratio * bar_w)
        
        # 漸層顏色
        if display_ratio < 0.3: 
            color = (0, 255, 255)  # 青色
        elif display_ratio < 0.6:
            color = (0, 255, 0)    # 綠色
        elif display_ratio < 0.8:
            color = (0, 165, 255)  # 橙色
        else:
            color = (255, 0, 0)    # 紅色
        
        # 繪製速度條
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + fill_w, start_y + bar_h), 
                     color, -1)
        
        # 邊框
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + bar_w, start_y + bar_h), 
                     (200, 200, 200), 2)
        
        # 添加刻度
        for i in range(1, 5):
            x_pos = start_x + int(i * 0.2 * bar_w)
            cv2.line(image, (x_pos, start_y), 
                    (x_pos, start_y + bar_h), (100, 100, 100), 1)
        
        # 標籤
        label_text = f"Speed: {display_val:.1f} m/s"
        cv2.putText(image, label_text, (start_x, start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 繪製骨架
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
            
            # 儲存歷史數據
            self.pos_history.append(landmarks)
            self.time_history.append(current_time)
            
            # === 狀態機邏輯 ===
            
            # 狀態 1: 等待預備姿勢
            if self.state == 'WAIT_GUARD':
                # 檢查姿勢是否正確
                guard_valid = self.check_guard_pose(landmarks)
                
                if guard_valid:
                    self.guard_stable_frames += 1
                    
                    if self.guard_stable_frames >= self.guard_stable_threshold:
                        self.guard_pose_valid = True
                        
                        # 計算保持時間進度
                        if self.guard_hold_start_time is None:
                            self.guard_hold_start_time = current_time
                        
                        hold_duration = current_time - self.guard_hold_start_time
                        required_duration = 1.5  # 需要保持1.5秒
                        progress = min(100, int((hold_duration / required_duration) * 100))
                        
                        # 繪製預備姿勢指示器
                        image = self.draw_guard_indicator(image, h, w, True, progress)
                        
                        # 如果保持足夠時間，進入預備狀態
                        if hold_duration > required_duration:
                            self.state = 'PRE_START'
                            self.wait_until = current_time + random.uniform(1.5, 3.0)
                            self.guard_hold_start_time = None
                            self.guard_stable_frames = 0
                    else:
                        progress = int((self.guard_stable_frames / self.guard_stable_threshold) * 100)
                        image = self.draw_guard_indicator(image, h, w, False, progress)
                else:
                    self.guard_stable_frames = 0
                    self.guard_hold_start_time = None
                    image = self.draw_guard_indicator(image, h, w, False, 0)
            
            # 狀態 2: 預備開始（倒數）
            elif self.state == 'PRE_START':
                # 持續檢查姿勢
                if not self.check_guard_pose(landmarks):
                    self.state = 'WAIT_GUARD'
                    image = self.draw_guard_indicator(image, h, w, False, 0)
                elif current_time > self.wait_until:
                    # 隨機選擇目標
                    self.state, self.target = 'STIMULUS', random.choice(['LEFT', 'RIGHT'])
                    self.start_time = current_time
                    self.command_display_until = current_time + 1.5  # 顯示1.5秒
                    
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
            
            # 狀態 3: 刺激階段（顯示提示）
            if self.state in ['STIMULUS', 'RESULT_PENDING']:
                # 顯示出拳提示（最前方）
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
                angle = self.calculate_angle(shoulder, elbow, wrist)
                
                # 計算速度
                velocity = 0
                acceleration = 0
                
                if self.prev_landmarks and dt > 0:
                    prev_wrist = self.prev_landmarks[wrist_idx]
                    
                    # 計算速度
                    total_v, forward_v = self.calculate_3d_velocity(wrist, prev_wrist, scale, dt)
                    velocity = forward_v  # 主要使用前向速度
                    
                    # 平滑處理
                    self.smoothed_speed = (self.smoothed_speed * 0.8 + velocity * 0.2)
                    
                    # 計算加速度
                    if self.prev_instant_v > 0 and dt > 0:
                        acceleration = (velocity - self.prev_instant_v) / dt
                    
                    self.prev_instant_v = velocity
                
                # 防誤觸機制：需要持續一定時間
                if velocity > self.MIN_VELOCITY_THRESHOLD:
                    if self.punch_start_time is None:
                        self.punch_start_time = current_time
                        self.punch_detection_active = True
                else:
                    if self.punch_detection_active and (current_time - self.punch_start_time < self.min_punch_duration):
                        # 短暫動作，忽略
                        self.false_trigger_count += 1
                        
                        if self.false_trigger_count >= self.false_trigger_threshold:
                            self.punch_detection_active = False
                            self.punch_start_time = None
                            self.false_trigger_count = 0
                
                # 更新最大值（只有在有效出拳檢測期間）
                if self.punch_detection_active:
                    self.max_v_temp = max(self.max_v_temp, self.smoothed_speed)
                    self.max_acc_temp = max(self.max_acc_temp, acceleration)
                    
                    if acceleration > self.MIN_ACCELERATION_THRESHOLD and self.acc_start_time is None:
                        self.acc_start_time = current_time
                
                # 擊中條件（更嚴格）
                cond_duration = (self.punch_start_time is not None and 
                                (current_time - self.punch_start_time) > self.min_punch_duration)
                cond_speed = self.max_v_temp > self.MIN_VELOCITY_THRESHOLD
                cond_acc = self.max_acc_temp > self.MIN_ACCELERATION_THRESHOLD
                cond_angle = angle > self.ARM_ANGLE_THRESHOLD
                cond_forward = (shoulder.z - wrist.z) > self.Z_PUNCH_THRESHOLD
                
                # 使用軌跡擬合計算最終速度
                final_speed = self.max_v_temp
                final_acc = self.max_acc_temp
                
                if self.punch_detection_active and len(self.pos_history) >= 5:
                    recent_positions = []
                    recent_times = []
                    
                    for i in range(min(8, len(self.pos_history))):
                        idx = -1 - i
                        pos = self.pos_history[idx]
                        wrist_pos = pos[wrist_idx]
                        recent_positions.append(wrist_pos)
                        recent_times.append(self.time_history[idx])
                    
                    recent_positions.reverse()
                    recent_times.reverse()
                    
                    _, peak_v, peak_a = self.calculate_speed_from_trajectory(recent_positions, recent_times, scale)
                    
                    if peak_v > 0:
                        final_speed = peak_v
                        final_acc = max(final_acc, peak_a)
                
                # 判定擊中（需要同時滿足多個條件）
                if (cond_duration and cond_speed and cond_acc and 
                    (cond_angle or cond_forward)):
                    
                    self.last_reaction_time = (current_time - self.start_time) * 1000
                    self.last_punch_speed = final_speed
                    self.last_punch_peak_acc = final_acc
                    
                    # 避免極端值
                    if self.last_punch_speed > 25.0:
                        self.last_punch_speed = min(25.0, final_speed)
                    
                    # 保存歷史數據
                    self.reaction_history.append(self.last_reaction_time)
                    self.speed_history.append(self.last_punch_speed)
                    self.acc_history.append(self.last_punch_peak_acc)
                    
                    self.show_results = True
                    self.state = 'RESULT_PENDING'
                    self.wait_until = current_time + 1.0
                
                # 超時處理
                if (current_time - self.start_time) > 4.0:
                    self.state = 'WAIT_GUARD'
            
            elif self.state == 'RESULT_PENDING':
                if current_time > self.wait_until:
                    self.state = 'RESULT'
                    self.wait_until = current_time + 2.0
            
            elif self.state == 'RESULT':
                if current_time > self.wait_until:
                    self.state = 'WAIT_GUARD'
                    self.guard_pose_valid = False
            
            self.prev_landmarks = landmarks
        
        self.prev_time = current_time
        
        # 繪製UI（分層繪製）
        if self.show_results and self.state != 'STIMULUS':
            image = self.draw_results(image, h, w)
        
        # 速度條總是顯示
        if self.state in ['STIMULUS', 'RESULT_PENDING', 'RESULT']:
            image = self.draw_speed_bar(image, h, w)
        
        # FPS顯示
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
    st.set_page_config(page_title="拳擊反應分析系統", layout="wide")
    st.title("🥊 拳擊反應分析系統 - 精準物理版")
    
    with st.sidebar:
        st.header("🎯 系統特色")
        st.success("**主要功能：**")
        st.write("✓ 強化預備姿勢檢測")
        st.write("✓ 防誤觸機制")
        st.write("✓ 分層UI設計")
        st.write("✓ 精準速度計算")
        
        st.divider()
        
        st.info("**使用說明：**")
        st.write("1. 面對鏡頭站立")
        st.write("2. 舉起雙手，拳頭放在臉頰兩側")
        st.write("3. 保持姿勢直到系統提示")
        st.write("4. 看到提示後快速出拳")
        st.write("5. 查看分析結果")
        
        st.divider()
        
        if st.button("🔄 重新開始"):
            st.experimental_rerun()
    
    # 主畫面
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("即時分析畫面")
        
        # 創建一個更寬容的媒體約束
        media_stream_constraints = {
            "video": {
                # 使用更寬容的設定
                "width": {"ideal": 640, "min": 320},
                "height": {"ideal": 480, "min": 240},
                "frameRate": {"ideal": 30, "min": 15}
            },
            "audio": False
        }
        
        # 可選：允許使用者選擇攝影機
        video_source = st.selectbox(
            "選擇攝影機",
            ["預設攝影機", "前置攝影機", "後置攝影機"],
            index=0
        )
        
        # 根據選擇調整約束
        if video_source == "前置攝影機":
            media_stream_constraints["video"]["facingMode"] = {"ideal": "user"}
        elif video_source == "後置攝影機":
            media_stream_constraints["video"]["facingMode"] = {"ideal": "environment"}
        
        ctx = webrtc_streamer(
            key="boxing-analyzer",
            video_processor_factory=VideoProcessor,
            media_stream_constraints=media_stream_constraints,
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        if not ctx.state.playing:
            st.info("👆 請點擊上方「START」按鈕開始")
            st.warning("如果遇到攝影機權限問題，請確保：")
            st.write("1. 允許瀏覽器存取攝影機")
            st.write("2. 沒有其他程式在使用攝影機")
            st.write("3. 刷新頁面重試")
    
    with col2:
        st.subheader("📊 等級對照表")
        
        with st.expander("速度等級", expanded=True):
            st.table({
                "等級": ["暖身", "初學者", "業餘", "專業", "選手", "世界級", "傳奇"],
                "速度(m/s)": ["<4", "4-6", "6-8", "8-10", "10-13", "13-16", ">16"],
            })
        
        with st.expander("反應時間", expanded=True):
            st.table({
                "等級": ["頂尖", "優異", "良好", "一般", "遲緩"],
                "時間(ms)": ["<120", "120-150", "150-200", "200-300", ">300"],
            })
        
        st.divider()
        
        st.info("**提示：**")
        st.write("- 確保良好光照")
        st.write("- 全身入鏡")
        st.write("- 出拳時動作明確")


if __name
