import cv2
import numpy as np
import streamlit as st
import time
import random
import math
import pandas as pd
from PIL import ImageFont, ImageDraw, Image # 新增 PIL 用於美觀字體

# 設置頁面
st.set_page_config(
    page_title="拳擊反應測試 (模擬版)",
    page_icon="🥊",
    layout="wide"
)

# 初始化 session state
if 'analyst' not in st.session_state:
    st.session_state.analyst = None
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
if 'results' not in st.session_state:
    st.session_state.results = {
        'reaction_history': [],
        'speed_history': [],
        'current_reaction': 0,
        'current_speed': 0,
        'test_count': 0
    }
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

class BoxingAnalyst:
    def __init__(self):
        # 狀態變數
        self.state = 'IDLE'  # IDLE, READY, COUNTDOWN, PUNCHING, RESULT
        self.target = None
        self.start_time = 0
        self.countdown_end = 0
        self.punch_detected = False
        self.punch_time = 0
        self.show_target = False
        self.target_start_time = 0
        
        # 速度計算
        self.current_speed = 0
        self.max_speed = 0
        
        # 物理參數
        self.MIN_PUNCH_SPEED = 2.0  
        
        # 模擬數據
        self.simulated_person = {
            'shoulders': [(0.3, 0.5), (0.7, 0.5)],
            'elbows': [(0.25, 0.65), (0.75, 0.65)],
            'wrists': [(0.2, 0.75), (0.8, 0.75)],
            'punching': False,
            'punch_progress': 0,
            'punch_side': None
        }

        # 字型設定 (同 v23)
        self.font_path = "font.ttf" 
        try:
            ImageFont.truetype(self.font_path, 20)
            self.use_chinese = True
        except:
            self.use_chinese = False

    def put_chinese_text(self, img, text, pos, color, size=30, stroke_width=0, stroke_fill=(0,0,0)):
        """ 繪製中文文字 (含描邊效果) """
        if not self.use_chinese:
            # OpenCV 使用 BGR
            cv2_color = (color[2], color[1], color[0]) 
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size/30, cv2_color, 2)
            return img
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, size)
        
        # 繪製文字 (含描邊)
        draw.text(pos, text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    def reset_test(self):
        """重置測試狀態"""
        self.state = 'IDLE'
        self.target = None
        self.start_time = 0
        self.countdown_end = 0
        self.punch_detected = False
        self.punch_time = 0
        self.show_target = False
        self.target_start_time = 0
        self.current_speed = 0
        self.max_speed = 0
        self.simulated_person['punching'] = False
        self.simulated_person['punch_progress'] = 0
        self.simulated_person['punch_side'] = None
        
    def start_test(self):
        """開始新測試"""
        self.reset_test()
        self.state = 'READY'
        self.target = random.choice(['LEFT', 'RIGHT'])
        self.start_time = time.time()
        
    def update_state(self):
        """更新狀態機"""
        current_time = time.time()
        
        if self.state == 'READY':
            # 準備1.5秒
            if current_time - self.start_time > 1.5:
                self.state = 'COUNTDOWN'
                self.countdown_end = current_time + random.uniform(0.8, 1.5)
                
        elif self.state == 'COUNTDOWN':
            if current_time > self.countdown_end:
                self.state = 'PUNCHING'
                self.start_time = current_time
                self.show_target = True
                self.target_start_time = current_time
                
        elif self.state == 'PUNCHING':
            # 如果1.5秒內沒出拳，超時
            if current_time - self.start_time > 1.5:
                self.state = 'RESULT'
                self.show_target = False
                
        elif self.state == 'RESULT':
            # 顯示結果2.5秒
            if current_time - self.start_time > 4.0:
                self.state = 'IDLE'
    
    def trigger_punch(self, side):
        """觸發出拳（手動）並生成模擬數據"""
        if self.state == 'PUNCHING' and side == self.target:
            current_time = time.time()
            
            self.simulated_person['punching'] = True
            self.simulated_person['punch_side'] = side
            self.simulated_person['punch_progress'] = 0
            
            # 計算反應時間
            self.punch_time = current_time
            self.punch_detected = True
            
            reaction_time = (self.punch_time - self.start_time) * 1000
            
            # === 修正：更新模擬速度生成的邏輯 (配合職業標準) ===
            # 越快反應，模擬出的速度越高
            if reaction_time < 150:
                # 職業級模擬
                base_speed = 13.0 + random.uniform(0, 4.0)  # 13-17 m/s
            elif reaction_time < 250:
                # 業餘/校隊模擬
                base_speed = 9.0 + random.uniform(0, 3.0)   # 9-12 m/s
            else:
                # 一般模擬
                base_speed = 5.0 + random.uniform(0, 3.0)   # 5-8 m/s
            
            variation = random.uniform(-0.5, 0.5)
            speed = base_speed + variation
            self.current_speed = max(self.MIN_PUNCH_SPEED, speed)
            self.max_speed = max(self.max_speed, self.current_speed)
            
            # 保存結果
            st.session_state.results['current_reaction'] = reaction_time
            st.session_state.results['current_speed'] = self.current_speed
            st.session_state.results['reaction_history'].append(reaction_time)
            st.session_state.results['speed_history'].append(self.current_speed)
            st.session_state.results['test_count'] += 1
            
            # 切換到結果狀態
            self.state = 'RESULT'
            self.show_target = False
            self.start_time = current_time
            
            return True
        return False
    
    def update_simulation(self):
        """更新模擬動畫"""
        if self.simulated_person['punching']:
            self.simulated_person['punch_progress'] += 0.15
            if self.simulated_person['punch_progress'] >= 1.0:
                self.simulated_person['punching'] = False
    
    def create_simulated_frame(self, width=640, height=480):
        """創建模擬畫面"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 60)  # 深藍灰色背景
        
        self.update_simulation()
        person = self.simulated_person
        
        left_wrist = list(person['wrists'][0])
        right_wrist = list(person['wrists'][1])
        
        if person['punching'] and person['punch_side']:
            progress = person['punch_progress']
            ease_progress = 1 - (1 - progress) ** 2  
            
            if person['punch_side'] == 'LEFT':
                left_wrist[0] = 0.2 - ease_progress * 0.25
                left_wrist[1] = 0.75 - ease_progress * 0.2 
            else:
                right_wrist[0] = 0.8 + ease_progress * 0.25
                right_wrist[1] = 0.75 - ease_progress * 0.2 
        
        def to_pixel(coord):
            return (int(coord[0] * width), int(coord[1] * height))
        
        # 繪製骨架 (保持 OpenCV 繪圖)
        color = (0, 255, 0)
        left_shoulder = to_pixel(person['shoulders'][0])
        right_shoulder = to_pixel(person['shoulders'][1])
        left_elbow = to_pixel(person['elbows'][0])
        right_elbow = to_pixel(person['elbows'][1])
        left_wrist_pixel = to_pixel(left_wrist)
        right_wrist_pixel = to_pixel(right_wrist)
        
        cv2.line(frame, left_shoulder, left_elbow, color, 3)
        cv2.line(frame, left_elbow, left_wrist_pixel, color, 3)
        cv2.line(frame, right_shoulder, right_elbow, color, 3)
        cv2.line(frame, right_elbow, right_wrist_pixel, color, 3)
        cv2.line(frame, left_shoulder, right_shoulder, color, 3)
        
        joint_radius = 6
        cv2.circle(frame, left_shoulder, joint_radius, (0, 0, 255), -1) 
        cv2.circle(frame, right_shoulder, joint_radius, (0, 0, 255), -1)
        cv2.circle(frame, left_elbow, joint_radius, (255, 0, 0), -1)  
        cv2.circle(frame, right_elbow, joint_radius, (255, 0, 0), -1)
        cv2.circle(frame, left_wrist_pixel, joint_radius, (0, 255, 255), -1) 
        cv2.circle(frame, right_wrist_pixel, joint_radius, (0, 255, 255), -1)
        
        # UI 層
        self.add_status_overlay(frame, width, height)
        
        if self.show_target:
            self.add_target_overlay(frame, width, height)
        
        if self.state == 'RESULT' and self.punch_detected:
            self.add_result_overlay(frame, width, height)
        
        return frame
    
    def add_status_overlay(self, frame, width, height):
        """添加狀態疊加層"""
        status_info = {
            'IDLE': ("準備開始", (255, 255, 255)),
            'READY': ("準備就緒", (0, 255, 255)),
            'COUNTDOWN': ("集中注意力...", (255, 255, 0)),
            'PUNCHING': ("出拳！", (0, 255, 0)),
            'RESULT': ("完成", (0, 255, 0))
        }
        
        status_text, status_color = status_info.get(self.state, ("未知", (255, 255, 255)))
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 使用新版文字繪製 (無描邊)
        frame = self.put_chinese_text(frame, f"狀態: {status_text}", (20, 20), status_color, 30)
        
        # 倒數計時
        if self.state == 'COUNTDOWN':
            remaining = max(0, self.countdown_end - time.time())
            countdown_text = f"{remaining:.1f}"
            
            # 中央大字體
            text_x = int(width/2) - 50
            text_y = int(height/3)
            if int(time.time() * 2) % 2 == 0:
                 frame = self.put_chinese_text(frame, countdown_text, (text_x, text_y), (255, 255, 0), 100, stroke_width=4)
        
        return frame # 記得回傳 frame
    
    def add_target_overlay(self, frame, width, height):
        """添加目標提示 (同步 v23 視覺效果)"""
        if not self.target:
            return
            
        target_text = "左拳！" if self.target == 'LEFT' else "右拳！"
        # v23 顏色標準: 左(青/Cyan), 右(紅/Red)
        target_color = (0, 255, 255) if self.target == 'LEFT' else (255, 50, 50)
        
        # 使用帶黑色邊框的大字體
        frame = self.put_chinese_text(
            frame, 
            target_text, 
            (int(width/2)-120, int(height/2)-50), 
            target_color, 
            size=100, 
            stroke_width=6, 
            stroke_fill=(0,0,0)
        )
        return frame
    
    def add_result_overlay(self, frame, width, height):
        """添加結果顯示 (同步 v23 評價標準)"""
        result_y = height - 220
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, result_y - 20), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        reaction = st.session_state.results['current_reaction']
        
        # 評價標準 (v23)
        if reaction < 120:
            rating = "👑 頂尖選手"
            rating_color = (0, 255, 255) # Cyan
        elif reaction < 250:
            rating = "🔥 優異"
            rating_color = (0, 255, 0)   # Green
        else:
            rating = "😐 一般"
            rating_color = (200, 200, 200) # Gray
        
        frame = self.put_chinese_text(frame, f"反應時間: {reaction:.0f} ms", (20, result_y + 30), (255, 255, 255), 30)
        frame = self.put_chinese_text(frame, f"評價: {rating}", (20, result_y + 70), rating_color, 30)
        
        speed = st.session_state.results['current_speed']
        
        # 拳速評級 (v23 職業標準)
        if speed >= 13.0:
            speed_rating = "💪 職業拳手"
            speed_color = (255, 50, 50) # Red
        elif speed >= 11.0:
            speed_rating = "🏆 選手級"
            speed_color = (255, 165, 0) # Orange
        elif speed >= 8.0:
            speed_rating = "🥊 校隊等級"
            speed_color = (255, 255, 0) # Yellow
        else:
            speed_rating = "🏃 慢速/暖身"
            speed_color = (150, 150, 150)
            
        frame = self.put_chinese_text(frame, f"出拳速度: {speed:.1f} m/s", (20, result_y + 120), (255, 255, 255), 30)
        frame = self.put_chinese_text(frame, f"等級: {speed_rating}", (20, result_y + 160), speed_color, 30)
        
        return frame

def main():
    st.title("🥊 拳擊反應測試系統 (模擬版)")
    
    with st.sidebar:
        st.header("使用說明")
        st.markdown("本版本為**模擬測試**，無需 Webcam。請使用下方按鈕或鍵盤進行反應測試。")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 開始測試", type="primary", use_container_width=True):
                if st.session_state.analyst is None: st.session_state.analyst = BoxingAnalyst()
                st.session_state.analyst.start_test()
                st.session_state.test_started = True
                st.session_state.last_update = time.time()
                st.rerun()
        with col2:
            if st.button("🔄 重置", type="secondary", use_container_width=True):
                if st.session_state.analyst: st.session_state.analyst.reset_test()
                st.session_state.results = {'reaction_history': [], 'speed_history': [], 'current_reaction': 0, 'current_speed': 0, 'test_count': 0}
                st.session_state.test_started = False
                st.rerun()
        
        st.divider()
        st.subheader("模擬出拳 (反應區)")
        col_left, col_right = st.columns(2)
        
        # 模擬按鈕
        with col_left:
            if st.button("👊 左拳", type="primary", use_container_width=True):
                if st.session_state.analyst and st.session_state.analyst.trigger_punch('LEFT'):
                    st.session_state.last_update = time.time()
                    st.rerun()
        with col_right:
            if st.button("👊 右拳", type="primary", use_container_width=True):
                if st.session_state.analyst and st.session_state.analyst.trigger_punch('RIGHT'):
                    st.session_state.last_update = time.time()
                    st.rerun()

    # 主畫面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        if st.session_state.analyst is None:
            st.session_state.analyst = BoxingAnalyst()
        analyst = st.session_state.analyst
        
        if st.session_state.test_started:
            analyst.update_state()
            current_time = time.time()
            st.session_state.frame_count += 1
            if current_time - st.session_state.last_update > 0.1:
                st.session_state.last_update = current_time
                st.rerun()
        
        frame = analyst.create_simulated_frame(width=800, height=600)
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
    with col2:
        st.subheader("即時數據 (專業版)")
        if st.session_state.test_started and analyst.punch_detected:
            # 使用 v23 顏色邏輯的 Metric
            reaction = st.session_state.results['current_reaction']
            speed = st.session_state.results['current_speed']
            
            st.metric("反應時間", f"{reaction:.0f} ms", delta="優異" if reaction < 250 else "一般")
            st.metric("拳速 (模擬)", f"{speed:.1f} m/s", delta="職業級" if speed > 13 else "普通")

        st.divider()
        st.markdown("### 歷史紀錄")
        if st.session_state.results['test_count'] > 0:
            df = pd.DataFrame({
                '次數': range(1, st.session_state.results['test_count']+1),
                '反應(ms)': st.session_state.results['reaction_history'],
                '速度(m/s)': st.session_state.results['speed_history']
            })
            st.dataframe(df.tail(5), hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
