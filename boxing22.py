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
from collections import deque

# ================= 配置與常數 =================
st.set_page_config(page_title="拳擊反應 v26 (進階版)", layout="wide", page_icon="🥊")

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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # 狀態機: WAIT_GUARD -> COUNTDOWN -> STIMULUS -> RESULT -> WAIT_NEXT
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
        self.punch_start_pos = None
        self.punch_end_pos = None
        
        # 計數器
        self.total_tests = 10
        self.current_test = 0
        self.guard_start_time = 0
        self.guard_required_time = 3.0  # 第一次需要3秒，之後2秒
        
        # 歷史數據 - 分左右手記錄
        self.left_history = {"reaction": [], "speed": []}
        self.right_history = {"reaction": [], "speed": []}
        self.last_result = {"reaction": 0, "speed": 0, "rating": "", "speed_rating": "", "hand": ""}
        
        # 測試結果摘要
        self.test_completed = False
        self.summary_data = {
            "left_avg_reaction": 0,
            "right_avg_reaction": 0,
            "left_avg_speed": 0,
            "right_avg_speed": 0,
            "left_rating": "",
            "right_rating": "",
            "overall_rating": ""
        }

        # 字型加載
        self.font_path = "font.ttf"
        self.use_chinese = False
        try:
            ImageFont.truetype(self.font_path, 20)
            self.use_chinese = True
        except:
            print("未找到字型檔，將使用預設字體")
            
        # 速度計算緩衝區
        self.speed_buffer = deque(maxlen=5)
        self.velocity_history = []

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

    def calculate_speed_advanced(self, current_coords, dt):
        """ 改進的拳速計算方法 """
        if not self.prev_landmarks or dt <= 0:
            return 0.0
            
        # 1. 計算像素比例尺
        shoulder_dist_px = np.linalg.norm(current_coords['L_SH'][:2] - current_coords['R_SH'][:2])
        if shoulder_dist_px < 10: 
            return 0.0
        
        pixels_per_meter = shoulder_dist_px / SHOULDER_WIDTH_M
        
        # 2. 判斷出拳手並記錄軌跡
        active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
        
        # 3. 計算3D位移
        curr_pos = current_coords[active_wrist]
        prev_pos = self.prev_landmarks[active_wrist]
        
        # 3D距離計算 (考慮深度)
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        dz = curr_pos[2] - prev_pos[2]
        dist_px = math.sqrt(dx*dx + dy*dy + dz*dz*0.3)  # z軸權重降低
        
        # 4. 計算即時速度
        speed_mps = (dist_px / pixels_per_meter) / dt
        
        # 5. 過濾和平滑
        self.speed_buffer.append(speed_mps)
        smoothed_speed = np.mean(self.speed_buffer)
        
        # 6. 保存速度歷史用於峰值檢測
        self.velocity_history.append(smoothed_speed)
        if len(self.velocity_history) > 20:
            self.velocity_history.pop(0)
        
        # 7. 物理合理性檢查
        if smoothed_speed > 25:  # 人類極限約20-22 m/s
            return 0.0
            
        return smoothed_speed

    def detect_punch_movement(self, coords):
        """ 檢測出拳動作 """
        if not coords:
            return False
            
        active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
        active_elbow = 'L_EL' if self.target == 'LEFT' else 'R_EL'
        
        # 1. 手肘角度檢測
        shoulder = coords['L_SH'] if self.target == 'LEFT' else coords['R_SH']
        elbow = coords[active_elbow]
        wrist = coords[active_wrist]
        
        # 計算手臂角度
        v1 = elbow[:2] - shoulder[:2]
        v2 = wrist[:2] - elbow[:2]
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return False
            
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.degrees(math.acos(cos_angle))
        
        # 出拳時手臂趨向伸直 (角度接近180度)
        if angle > 150:  # 手臂較直
            # 2. 方向一致性檢測 (手臂向前移動)
            if self.prev_landmarks:
                wrist_movement = wrist[:2] - self.prev_landmarks[active_wrist][:2]
                # 檢查是否向前移動 (假設鏡頭方向)
                if wrist_movement[0] > 0:  # 向右移動 (因為畫面鏡像)
                    return True
                    
        return False

    def process(self, img):
        # 1. 影像前處理
        img = cv2.flip(img, 1)  # 鏡像
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

        # 顯示測試進度
        if not self.test_completed:
            progress_text = f"測試進度: {self.current_test}/{self.total_tests}"
            img = self.put_chinese_text(img, progress_text, (w-250, 30), (255, 255, 255), 30)

        # 狀態機邏輯
        if self.state == 'WAIT_GUARD':
            # 顯示指引
            if self.current_test == 0:
                req_time = 3.0
            else:
                req_time = 2.0
                
            img = self.put_chinese_text(img, f"請擺出格鬥姿勢 ({req_time}秒後開始)", (20, 50), COLOR_TEXT, 40, stroke_width=2)
            
            if coords:
                # 判定防禦姿勢
                l_guard = coords['L_WR'][1] < coords['L_EL'][1]
                r_guard = coords['R_WR'][1] < coords['R_EL'][1]
                
                if l_guard and r_guard:
                    cv2.rectangle(img, (0,0), (w, h), (0, 255, 0), 5)  # 綠框提示
                    
                    if self.guard_start_time == 0:
                        self.guard_start_time = current_time
                    
                    elapsed = current_time - self.guard_start_time
                    remaining = max(0, req_time - elapsed)
                    
                    # 顯示倒數
                    img = self.put_chinese_text(img, f"{remaining:.1f}秒", (w//2-50, 100), (0, 255, 255), 50)
                    
                    if elapsed >= req_time:
                        self.state = 'COUNTDOWN'
                        self.start_time = current_time
                        self.guard_start_time = 0
                else:
                    self.guard_start_time = 0  # 重置計時

        elif self.state == 'COUNTDOWN':
            remaining = 3.0 - (current_time - self.start_time)
            if remaining <= 0:
                self.state = 'STIMULUS'
                self.target = random.choice(['LEFT', 'RIGHT'])
                self.stimulus_time = current_time
                self.max_speed = 0
                self.speed_buffer.clear()
                self.velocity_history.clear()
                self.punch_start_pos = None
                self.punch_end_pos = None
            else:
                # 中央倒數
                cx, cy = int(w/2)-20, int(h/2)
                img = self.put_chinese_text(img, f"{int(remaining)+1}", (cx, cy), (0, 255, 255), 100, stroke_width=4)

        elif self.state == 'STIMULUS':
            # 顯示視覺刺激
            text = "左拳!" if self.target == 'LEFT' else "右拳!"
            color = COLOR_CYAN if self.target == 'LEFT' else COLOR_RED
            
            cx, cy = int(w/2)-100, int(h/2)
            img = self.put_chinese_text(img, text, (cx, cy), color, 120, stroke_width=6)
            
            # 偵測出拳
            if coords and self.prev_landmarks:
                speed = self.calculate_speed_advanced(coords, dt)
                
                # 記錄出拳開始位置
                if self.punch_start_pos is None and speed > 1.0:
                    active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
                    self.punch_start_pos = coords[active_wrist].copy()
                
                # 更新最大速度
                if speed > self.max_speed:
                    self.max_speed = speed
                
                # 觸發條件：速度峰值 + 手臂伸直
                if speed > 4.0 and self.detect_punch_movement(coords):
                    self.state = 'RESULT'
                    reaction_time = (current_time - self.stimulus_time) * 1000
                    
                    # 記錄出拳結束位置
                    active_wrist = 'L_WR' if self.target == 'LEFT' else 'R_WR'
                    self.punch_end_pos = coords[active_wrist].copy()
                    
                    # 計算最終速度（使用峰值速度）
                    if len(self.velocity_history) > 3:
                        final_speed = np.max(self.velocity_history[-5:])  # 使用最近5幀的最大值
                    else:
                        final_speed = self.max_speed
                    
                    # 更新最後結果
                    self.last_result['reaction'] = reaction_time
                    self.last_result['speed'] = final_speed
                    self.last_result['hand'] = self.target
                    
                    # 評價邏輯
                    if reaction_time < 120: 
                        self.last_result['rating'] = "👑 頂尖"
                    elif reaction_time < 250: 
                        self.last_result['rating'] = "🔥 優異"
                    else: 
                        self.last_result['rating'] = "😐 一般"
                    
                    if final_speed > 13: 
                        self.last_result['speed_rating'] = "💪 職業級"
                    elif final_speed > 9: 
                        self.last_result['speed_rating'] = "🏆 選手級"
                    else: 
                        self.last_result['speed_rating'] = "🏃 業餘"
                    
                    # 記錄到歷史數據
                    if self.target == 'LEFT':
                        self.left_history["reaction"].append(reaction_time)
                        self.left_history["speed"].append(final_speed)
                    else:
                        self.right_history["reaction"].append(reaction_time)
                        self.right_history["speed"].append(final_speed)
                    
                    self.current_test += 1
                    self.feedback_end_time = current_time + 2.5  # 顯示結果2.5秒

        elif self.state == 'RESULT':
            # 顯示單次結果
            rt = self.last_result['reaction']
            sp = self.last_result['speed']
            hand = self.last_result['hand']
            
            # 繪製結果面板
            overlay = img.copy()
            cv2.rectangle(overlay, (50, h-250), (450, h-50), (0,0,0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            hand_color = COLOR_CYAN if hand == 'LEFT' else COLOR_RED
            hand_text = "左拳" if hand == 'LEFT' else "右拳"
            
            img = self.put_chinese_text(img, f"{hand_text} 結果", (70, h-220), hand_color, 35)
            img = self.put_chinese_text(img, f"反應: {rt:.0f} ms", (70, h-180), (255, 255, 255), 30)
            img = self.put_chinese_text(img, f"評價: {self.last_result['rating']}", (70, h-145), hand_color, 30)
            img = self.put_chinese_text(img, f"拳速: {sp:.1f} m/s", (70, h-110), (255, 255, 255), 30)
            img = self.put_chinese_text(img, f"等級: {self.last_result['speed_rating']}", (70, h-75), hand_color, 30)
            
            # 顯示下一個提示
            if self.current_test < self.total_tests:
                next_text = f"準備下一拳 ({2 if self.current_test > 0 else 3}秒預備)"
                img = self.put_chinese_text(img, next_text, (w//2-200, 150), (255, 255, 0), 40)
            
            if current_time > self.feedback_end_time:
                if self.current_test >= self.total_tests:
                    self.calculate_summary()
                    self.state = 'SUMMARY'
                else:
                    self.state = 'WAIT_GUARD'
                    self.guard_start_time = 0

        elif self.state == 'SUMMARY':
            # 計算並顯示最終結果
            self.display_summary(img, w, h)

        # 更新上一幀座標
        self.prev_landmarks = coords
        return img

    def calculate_summary(self):
        """ 計算最終統計數據 """
        self.test_completed = True
        
        # 計算平均值
        if self.left_history["reaction"]:
            self.summary_data["left_avg_reaction"] = np.mean(self.left_history["reaction"])
            self.summary_data["left_avg_speed"] = np.mean(self.left_history["speed"])
        
        if self.right_history["reaction"]:
            self.summary_data["right_avg_reaction"] = np.mean(self.right_history["reaction"])
            self.summary_data["right_avg_speed"] = np.mean(self.right_history["speed"])
        
        # 評價邏輯
        def get_rating(reaction, speed):
            rating = []
            if reaction < 150:
                rating.append("頂尖反應")
            elif reaction < 280:
                rating.append("良好反應")
            else:
                rating.append("普通反應")
                
            if speed > 12:
                rating.append("職業拳速")
            elif speed > 8:
                rating.append("選手拳速")
            else:
                rating.append("業餘拳速")
                
            return " | ".join(rating)
        
        # 左右手評價
        if self.left_history["reaction"]:
            self.summary_data["left_rating"] = get_rating(
                self.summary_data["left_avg_reaction"], 
                self.summary_data["left_avg_speed"]
            )
        
        if self.right_history["reaction"]:
            self.summary_data["right_rating"] = get_rating(
                self.summary_data["right_avg_reaction"], 
                self.summary_data["right_avg_speed"]
            )
        
        # 整體評價
        all_reactions = self.left_history["reaction"] + self.right_history["reaction"]
        all_speeds = self.left_history["speed"] + self.right_history["speed"]
        
        if all_reactions:
            avg_reaction = np.mean(all_reactions)
            avg_speed = np.mean(all_speeds)
            
            if avg_reaction < 160 and avg_speed > 10:
                self.summary_data["overall_rating"] = "🎯 優秀拳擊手潛質"
            elif avg_reaction < 200 and avg_speed > 7:
                self.summary_data["overall_rating"] = "⭐ 良好運動能力"
            else:
                self.summary_data["overall_rating"] = "💪 持續練習可進步"

    def display_summary(self, img, w, h):
        """ 顯示最終結果面板 """
        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
        
        # 標題
        img = self.put_chinese_text(img, "🎯 測試完成！", (w//2-150, 100), (255, 255, 0), 60, stroke_width=3)
        
        # 左拳結果
        left_y = 200
        if self.left_history["reaction"]:
            img = self.put_chinese_text(img, "🥊 左拳統計", (w//4-100, left_y), COLOR_CYAN, 45)
            img = self.put_chinese_text(img, f"平均反應: {self.summary_data['left_avg_reaction']:.0f} ms", 
                                       (w//4-150, left_y+60), (255, 255, 255), 35)
            img = self.put_chinese_text(img, f"平均拳速: {self.summary_data['left_avg_speed']:.1f} m/s", 
                                       (w//4-150, left_y+110), (255, 255, 255), 35)
            img = self.put_chinese_text(img, f"評價: {self.summary_data['left_rating']}", 
                                       (w//4-150, left_y+160), COLOR_CYAN, 30)
        else:
            img = self.put_chinese_text(img, "左拳: 未測試", (w//4-100, left_y), (100, 100, 100), 40)
        
        # 右拳結果
        right_y = 200
        if self.right_history["reaction"]:
            img = self.put_chinese_text(img, "🥊 右拳統計", (3*w//4-100, right_y), COLOR_RED, 45)
            img = self.put_chinese_text(img, f"平均反應: {self.summary_data['right_avg_reaction']:.0f} ms", 
                                       (3*w//4-150, right_y+60), (255, 255, 255), 35)
            img = self.put_chinese_text(img, f"平均拳速: {self.summary_data['right_avg_speed']:.1f} m/s", 
                                       (3*w//4-150, right_y+110), (255, 255, 255), 35)
            img = self.put_chinese_text(img, f"評價: {self.summary_data['right_rating']}", 
                                       (3*w//4-150, right_y+160), COLOR_RED, 30)
        else:
            img = self.put_chinese_text(img, "右拳: 未測試", (3*w//4-100, right_y), (100, 100, 100), 40)
        
        # 整體評價
        if self.summary_data["overall_rating"]:
            img = self.put_chinese_text(img, "📋 整體評價", (w//2-100, h-250), (255, 255, 255), 50)
            img = self.put_chinese_text(img, self.summary_data["overall_rating"], 
                                       (w//2-200, h-180), (0, 255, 255), 40)
        
        # 操作提示
        img = self.put_chinese_text(img, "請查看右側面板選擇下一步", (w//2-200, h-80), (200, 200, 255), 30)

# ================= Streamlit 介面 =================
def main():
    st.title("🥊 拳擊反應測試 v26 (10次測驗版)")
    
    # 初始化session state
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'restart_flag' not in st.session_state:
        st.session_state.restart_flag = False
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # WebRTC 串流設定
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_streamer(
            key="boxing-pro-advanced",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.header("📊 測試控制面板")
        
        if not st.session_state.test_started:
            if st.button("🚀 開始10次測驗", use_container_width=True, type="primary"):
                st.session_state.test_started = True
                st.session_state.test_completed = False
                st.session_state.restart_flag = True
                st.rerun()
        
        st.divider()
        
        if st.session_state.test_started:
            st.subheader("測試說明")
            st.markdown("""
            **測試流程:**
            1. 第1次: 維持預備姿勢 **3秒**
            2. 第2-10次: 維持預備姿勢 **2秒**
            3. 看到文字提示後立即出拳
            4. 完成10次後顯示統計結果
            """)
            
            st.divider()
            
            if st.session_state.test_completed:
                st.success("✅ 測試已完成！")
                
                col_restart, col_exit = st.columns(2)
                with col_restart:
                    if st.button("🔄 重新測驗", use_container_width=True):
                        st.session_state.test_started = True
                        st.session_state.test_completed = False
                        st.session_state.restart_flag = True
                        st.rerun()
                
                with col_exit:
                    if st.button("🏁 結束測驗", use_container_width=True):
                        st.session_state.test_started = False
                        st.session_state.test_completed = False
                        st.rerun()
            
            st.divider()
            st.markdown("### 🎯 評價標準")
            st.caption("**反應時間:**")
            st.caption("- <150ms: 頂尖反應")
            st.caption("- 150-280ms: 良好反應")
            st.caption("- >280ms: 普通反應")
            
            st.caption("**出拳速度:**")
            st.caption("- >12m/s: 職業級")
            st.caption("- 8-12m/s: 選手級")
            st.caption("- <8m/s: 業餘級")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.logic = BoxingAnalystLogic()
        self.last_restart_flag = False
    
    def recv(self, frame):
        try:
            # 檢查是否需要重啟
            if hasattr(st.session_state, 'restart_flag') and st.session_state.restart_flag != self.last_restart_flag:
                self.logic = BoxingAnalystLogic()  # 重新初始化邏輯
                self.last_restart_flag = st.session_state.restart_flag
                st.session_state.restart_flag = False
            
            img = frame.to_ndarray(format="bgr24")
            processed_img = self.logic.process(img)
            
            # 更新測試完成狀態
            if self.logic.test_completed and not st.session_state.test_completed:
                st.session_state.test_completed = True
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

if __name__ == "__main__":
    main()
