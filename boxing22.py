import cv2
import numpy as np
import streamlit as st
import time
import random
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from collections import deque
import math

# 設置頁面
st.set_page_config(
    page_title="拳擊反應測試",
    page_icon="🥊",
    layout="wide"
)

# 初始化 session state
if 'analyst' not in st.session_state:
    st.session_state.analyst = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
if 'results' not in st.session_state:
    st.session_state.results = {
        'reaction_history': [],
        'speed_history': [],
        'current_reaction': 0,
        'current_speed': 0
    }

class BoxingAnalyst:
    def __init__(self):
        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # 狀態變數
        self.state = 'IDLE'  # IDLE, READY, COUNTDOWN, PUNCHING, RESULT
        self.target = None
        self.start_time = 0
        self.countdown_end = 0
        self.punch_detected = False
        self.punch_time = 0
        
        # 速度計算
        self.prev_positions = {}
        self.prev_time = 0
        self.current_speed = 0
        
        # 物理參數
        self.SHOULDER_WIDTH = 0.45  # 平均肩寬（米）
        
    def reset_test(self):
        """重置測試狀態"""
        self.state = 'IDLE'
        self.target = None
        self.start_time = 0
        self.countdown_end = 0
        self.punch_detected = False
        self.punch_time = 0
        self.current_speed = 0
        
    def start_test(self):
        """開始新測試"""
        self.reset_test()
        self.state = 'READY'
        self.target = random.choice(['LEFT', 'RIGHT'])
        
    def update_state(self):
        """更新狀態機"""
        current_time = time.time()
        
        if self.state == 'READY':
            # 等待2秒後開始倒數
            if current_time - self.start_time > 2:
                self.state = 'COUNTDOWN'
                self.countdown_end = current_time + random.uniform(1.0, 2.5)
                
        elif self.state == 'COUNTDOWN':
            if current_time > self.countdown_end:
                self.state = 'PUNCHING'
                self.start_time = current_time
                
        elif self.state == 'PUNCHING':
            # 如果3秒內沒出拳，超時
            if current_time - self.start_time > 3:
                self.state = 'RESULT'
                
        elif self.state == 'RESULT':
            # 顯示結果2秒
            if current_time - self.start_time > 5:
                self.state = 'IDLE'
    
    def calculate_speed(self, wrist_pos, prev_wrist_pos, dt):
        """計算拳速"""
        if dt <= 0 or prev_wrist_pos is None:
            return 0
            
        # 計算位移（使用Z軸為主）
        dz = prev_wrist_pos[2] - wrist_pos[2]  # MediaPipe: Z越小越近
        
        # 轉換為實際距離（米）
        distance = abs(dz) * self.SHOULDER_WIDTH
        
        # 計算速度（米/秒）
        speed = distance / dt if dt > 0 else 0
        
        return speed
    
    def detect_punch(self, landmarks):
        """檢測出拳"""
        if not landmarks:
            return False
            
        # 根據目標選擇手腕
        if self.target == 'LEFT':
            wrist = landmarks[15]  # 左手腕
            elbow = landmarks[13]  # 左手肘
            shoulder = landmarks[11]  # 左肩
        else:
            wrist = landmarks[16]  # 右手腕
            elbow = landmarks[14]  # 右手肘
            shoulder = landmarks[12]  # 右肩
            
        # 計算手臂角度
        def calculate_angle(a, b, c):
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        
        try:
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # 出拳條件：手臂較直（角度>120度）且手腕在肩膀前方
            is_extended = angle > 120
            is_forward = wrist.z < shoulder.z - 0.1
            
            return is_extended and is_forward
        except:
            return False
    
    def process_frame(self, frame):
        """處理影片幀"""
        # 轉換為RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿勢檢測
        results = self.pose.process(rgb_frame)
        
        # 更新狀態
        self.update_state()
        
        # 繪製結果
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]
        
        # 繪製狀態信息
        self.draw_status(output_frame, h, w)
        
        if results.pose_landmarks:
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # 檢測出拳
            if self.state == 'PUNCHING' and not self.punch_detected:
                if self.detect_punch(landmarks):
                    self.punch_detected = True
                    self.punch_time = time.time()
                    
                    # 計算反應時間
                    reaction_time = (self.punch_time - self.start_time) * 1000  # 轉為毫秒
                    
                    # 計算速度
                    current_time = time.time()
                    dt = current_time - self.prev_time
                    
                    if self.target == 'LEFT':
                        wrist_idx = 15
                    else:
                        wrist_idx = 16
                        
                    wrist = landmarks[wrist_idx]
                    wrist_pos = (wrist.x, wrist.y, wrist.z)
                    
                    if wrist_idx in self.prev_positions and dt > 0:
                        speed = self.calculate_speed(wrist_pos, self.prev_positions[wrist_idx], dt)
                        self.current_speed = speed
                        
                        # 保存結果
                        st.session_state.results['current_reaction'] = reaction_time
                        st.session_state.results['current_speed'] = speed
                        st.session_state.results['reaction_history'].append(reaction_time)
                        st.session_state.results['speed_history'].append(speed)
                    
                    self.prev_positions[wrist_idx] = wrist_pos
                    self.prev_time = current_time
                    
                    # 切換到結果狀態
                    self.state = 'RESULT'
                    self.start_time = time.time()
            
            # 保存當前位置用於速度計算
            current_time = time.time()
            dt = current_time - self.prev_time
            
            if dt > 0.033:  # 約30fps
                if self.target:
                    if self.target == 'LEFT':
                        wrist_idx = 15
                    else:
                        wrist_idx = 16
                        
                    wrist = landmarks[wrist_idx]
                    wrist_pos = (wrist.x, wrist.y, wrist.z)
                    self.prev_positions[wrist_idx] = wrist_pos
                    self.prev_time = current_time
        
        return output_frame
    
    def draw_status(self, frame, h, w):
        """繪製狀態信息"""
        # 狀態文字和顏色
        status_info = {
            'IDLE': ("準備開始", (255, 255, 255)),
            'READY': ("準備就緒", (0, 255, 255)),
            'COUNTDOWN': ("準備出拳...", (255, 255, 0)),
            'PUNCHING': ("出拳！", (0, 255, 0)),
            'RESULT': ("完成", (255, 0, 0))
        }
        
        status_text, status_color = status_info.get(self.state, ("未知", (255, 255, 255)))
        
        # 繪製狀態框
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"狀態: {status_text}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 顯示目標
        if self.target:
            target_text = "目標: 左拳" if self.target == 'LEFT' else "目標: 右拳"
            target_color = (0, 255, 255) if self.target == 'LEFT' else (255, 0, 255)
            cv2.putText(frame, target_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)
        
        # 顯示倒數
        if self.state == 'COUNTDOWN':
            remaining = max(0, self.countdown_end - time.time())
            countdown_text = f"倒數: {remaining:.1f}s"
            cv2.putText(frame, countdown_text, (w - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        
        # 顯示結果
        if self.state == 'RESULT' and self.punch_detected:
            reaction = st.session_state.results['current_reaction']
            speed = st.session_state.results['current_speed']
            
            result_y = h - 150
            cv2.putText(frame, f"反應時間: {reaction:.0f} ms", 
                       (20, result_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"出拳速度: {speed:.1f} m/s", 
                       (20, result_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 評價
            if reaction < 150:
                rating = "優異！"
                rating_color = (0, 255, 0)
            elif reaction < 250:
                rating = "良好"
                rating_color = (255, 255, 0)
            else:
                rating = "加油"
                rating_color = (255, 0, 0)
                
            cv2.putText(frame, f"評價: {rating}", 
                       (20, result_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, rating_color, 2)

# 主應用
def main():
    st.title("🥊 拳擊反應測試系統")
    
    # 側邊欄
    with st.sidebar:
        st.header("使用說明")
        st.markdown("""
        1. **點擊『開始測試』按鈕**
        2. **面對鏡頭站立**
        3. **看到『出拳！』提示後快速出拳**
        4. **查看你的反應時間和拳速**
        """)
        
        st.divider()
        
        if st.button("🔄 開始測試", type="primary", use_container_width=True):
            if st.session_state.analyst is None:
                st.session_state.analyst = BoxingAnalyst()
            st.session_state.analyst.start_test()
            st.session_state.test_started = True
            st.rerun()
            
        if st.button("🔄 重置數據", type="secondary", use_container_width=True):
            st.session_state.results = {
                'reaction_history': [],
                'speed_history': [],
                'current_reaction': 0,
                'current_speed': 0
            }
            if st.session_state.analyst:
                st.session_state.analyst.reset_test()
            st.rerun()
        
        st.divider()
        
        # 顯示統計數據
        st.subheader("測試統計")
        if st.session_state.results['reaction_history']:
            avg_reaction = np.mean(st.session_state.results['reaction_history'])
            avg_speed = np.mean(st.session_state.results['speed_history'])
            best_reaction = min(st.session_state.results['reaction_history'])
            best_speed = max(st.session_state.results['speed_history'])
            
            st.metric("測試次數", len(st.session_state.results['reaction_history']))
            st.metric("平均反應時間", f"{avg_reaction:.0f} ms")
            st.metric("平均拳速", f"{avg_speed:.1f} m/s")
            st.metric("最佳反應", f"{best_reaction:.0f} ms")
            st.metric("最快拳速", f"{best_speed:.1f} m/s")
        else:
            st.info("尚未進行測試")
        
        st.divider()
        
        st.subheader("評分標準")
        st.markdown("""
        **反應時間：**
        - < 150 ms: 🥇 優異
        - 150-250 ms: 🥈 良好
        - > 250 ms: 🥉 加油
        
        **拳速：**
        - > 8 m/s: 💪 專業級
        - 5-8 m/s: 👍 業餘級
        - < 5 m/s: 👊 初學級
        """)
    
    # 主內容區
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("即時分析")
        
        # 攝影機選擇
        camera_option = st.selectbox(
            "選擇攝影機",
            ["使用範例影片", "使用網路攝影機"],
            index=0,
            help="選擇『使用範例影片』進行演示，或選擇『使用網路攝影機』使用你的攝影機"
        )
        
        # 創建影片顯示區域
        video_placeholder = st.empty()
        
        if camera_option == "使用範例影片":
            # 使用範例影片
            st.info("使用範例影片進行演示。請舉起雙手模擬出拳動作。")
            
            # 載入範例影片
            cap = cv2.VideoCapture(0)  # 使用第一個攝影機作為範例
            
            if not cap.isOpened():
                # 創建一個簡單的測試影片
                st.warning("無法開啟攝影機，使用測試畫面")
                
                # 創建測試畫面
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "測試模式", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                if st.session_state.analyst and st.session_state.test_started:
                    processed_frame = st.session_state.analyst.process_frame(frame)
                    video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                else:
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)
            else:
                # 處理攝影機影片
                while st.session_state.test_started:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 處理幀
                    if st.session_state.analyst:
                        processed_frame = st.session_state.analyst.process_frame(frame)
                        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                    else:
                        video_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # 控制幀率
                    time.sleep(0.033)  # 約30fps
                
                cap.release()
                
        else:
            # 使用網路攝影機
            st.info("請允許瀏覽器存取攝影機權限")
            
            # 使用 streamlit 的 camera_input
            img_file_buffer = st.camera_input("開啟你的攝影機")
            
            if img_file_buffer is not None:
                # 讀取圖片
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # 處理圖片
                if st.session_state.analyst and st.session_state.test_started:
                    processed_img = st.session_state.analyst.process_frame(cv2_img)
                    video_placeholder.image(processed_img, channels="BGR", use_column_width=True)
                else:
                    video_placeholder.image(cv2_img, channels="BGR", use_column_width=True)
            else:
                # 顯示等待畫面
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "等待攝影機...", (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
    
    with col2:
        st.subheader("即時數據")
        
        # 當前測試數據
        if st.session_state.test_started and st.session_state.analyst:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    "當前狀態",
                    st.session_state.analyst.state,
                    delta=None
                )
                
            with col_b:
                if st.session_state.analyst.target:
                    target_text = "左拳" if st.session_state.analyst.target == 'LEFT' else "右拳"
                    st.metric("目標", target_text)
            
            # 速度顯示
            st.progress(
                min(1.0, st.session_state.analyst.current_speed / 15.0),
                text=f"拳速: {st.session_state.analyst.current_speed:.1f} m/s"
            )
            
            # 反應時間顯示
            if st.session_state.results['current_reaction'] > 0:
                reaction = st.session_state.results['current_reaction']
                st.progress(
                    min(1.0, 1.0 - (reaction / 500.0)),  # 500ms為最慢
                    text=f"反應時間: {reaction:.0f} ms"
                )
        
        st.divider()
        
        # 歷史數據圖表
        st.subheader("歷史表現")
        
        if st.session_state.results['reaction_history']:
            import pandas as pd
            
            # 創建數據框
            history_data = pd.DataFrame({
                '測試次數': range(1, len(st.session_state.results['reaction_history']) + 1),
                '反應時間(ms)': st.session_state.results['reaction_history'],
                '拳速(m/s)': st.session_state.results['speed_history']
            })
            
            # 顯示表格
            st.dataframe(
                history_data,
                use_container_width=True,
                hide_index=True
            )
            
            # 趨勢圖
            st.line_chart(history_data.set_index('測試次數'))
        else:
            st.info("尚未有測試數據")
        
        st.divider()
        
        # 使用提示
        st.info("💡 **提示**")
        st.markdown("""
        - 確保良好照明
        - 全身入鏡
        - 出拳動作要明確
        - 保持放鬆，反應更快
        """)

if __name__ == "__main__":
    main()
