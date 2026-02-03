import cv2
import numpy as np
import streamlit as st
import time
import random
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
        self.SHOULDER_WIDTH = 0.45  # 平均肩寬（米）
        self.MIN_PUNCH_SPEED = 2.0  # 最小出拳速度
        
        # 模擬數據
        self.simulated_person = {
            'shoulders': [(0.3, 0.5), (0.7, 0.5)],
            'elbows': [(0.25, 0.65), (0.75, 0.65)],
            'wrists': [(0.2, 0.75), (0.8, 0.75)],
            'punching': False,
            'punch_progress': 0,
            'punch_side': None
        }
        
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
        """觸發出拳（手動）"""
        if self.state == 'PUNCHING' and side == self.target:
            current_time = time.time()
            
            self.simulated_person['punching'] = True
            self.simulated_person['punch_side'] = side
            self.simulated_person['punch_progress'] = 0
            
            # 計算反應時間
            self.punch_time = current_time
            self.punch_detected = True
            
            reaction_time = (self.punch_time - self.start_time) * 1000
            
            # 計算速度（根據反應時間生成合理的速度）
            # 反應越快，速度越高
            if reaction_time < 150:
                base_speed = 8.0 + random.uniform(0, 3.0)  # 8-11 m/s
            elif reaction_time < 250:
                base_speed = 6.0 + random.uniform(0, 2.0)  # 6-8 m/s
            else:
                base_speed = 4.0 + random.uniform(0, 2.0)  # 4-6 m/s
            
            # 添加隨機變化
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
        
        # 更新模擬動畫
        self.update_simulation()
        
        # 繪製模擬人物
        person = self.simulated_person
        
        # 根據目標和狀態更新手腕位置
        left_wrist = list(person['wrists'][0])
        right_wrist = list(person['wrists'][1])
        
        if person['punching'] and person['punch_side']:
            progress = person['punch_progress']
            ease_progress = 1 - (1 - progress) ** 2  # 緩入緩出
            
            if person['punch_side'] == 'LEFT':
                # 左拳向前
                left_wrist[0] = 0.2 - ease_progress * 0.25  # 向左移動
                left_wrist[1] = 0.75 - ease_progress * 0.2  # 向上移動
            else:
                # 右拳向前
                right_wrist[0] = 0.8 + ease_progress * 0.25  # 向右移動
                right_wrist[1] = 0.75 - ease_progress * 0.2  # 向上移動
        
        # 轉換為像素座標
        def to_pixel(coord):
            x, y = coord
            return (int(x * width), int(y * height))
        
        # 繪製骨架
        color = (0, 255, 0)  # 綠色
        
        # 肩膀
        left_shoulder = to_pixel(person['shoulders'][0])
        right_shoulder = to_pixel(person['shoulders'][1])
        
        # 手肘
        left_elbow = to_pixel(person['elbows'][0])
        right_elbow = to_pixel(person['elbows'][1])
        
        # 手腕
        left_wrist_pixel = to_pixel(left_wrist)
        right_wrist_pixel = to_pixel(right_wrist)
        
        # 繪製線條（骨架）
        # 左臂
        cv2.line(frame, left_shoulder, left_elbow, color, 3)
        cv2.line(frame, left_elbow, left_wrist_pixel, color, 3)
        
        # 右臂
        cv2.line(frame, right_shoulder, right_elbow, color, 3)
        cv2.line(frame, right_elbow, right_wrist_pixel, color, 3)
        
        # 肩膀連線
        cv2.line(frame, left_shoulder, right_shoulder, color, 3)
        
        # 繪製關節點
        joint_radius = 6
        cv2.circle(frame, left_shoulder, joint_radius, (0, 0, 255), -1)  # 紅色
        cv2.circle(frame, right_shoulder, joint_radius, (0, 0, 255), -1)
        cv2.circle(frame, left_elbow, joint_radius, (255, 0, 0), -1)  # 藍色
        cv2.circle(frame, right_elbow, joint_radius, (255, 0, 0), -1)
        cv2.circle(frame, left_wrist_pixel, joint_radius, (0, 255, 255), -1)  # 黃色
        cv2.circle(frame, right_wrist_pixel, joint_radius, (0, 255, 255), -1)
        
        # 添加狀態文字
        self.add_status_overlay(frame, width, height)
        
        # 添加目標提示
        if self.show_target:
            self.add_target_overlay(frame, width, height)
        
        # 添加結果顯示
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
            'RESULT': ("完成", (255, 0, 0))
        }
        
        status_text, status_color = status_info.get(self.state, ("未知", (255, 255, 255)))
        
        # 狀態框
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (10, 10), (300, 80), status_color, 2)
        
        cv2.putText(frame, f"狀態: {status_text}", 
                   (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 倒數計時
        if self.state == 'COUNTDOWN':
            remaining = max(0, self.countdown_end - time.time())
            countdown_text = f"{remaining:.1f}"
            
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 3
            
            # 閃爍效果
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 4)
    
    def add_target_overlay(self, frame, width, height):
        """添加目標提示"""
        if not self.target:
            return
            
        target_text = "左拳！" if self.target == 'LEFT' else "右拳！"
        target_color = (0, 200, 255) if self.target == 'LEFT' else (255, 50, 150)
        
        # 大文字提示
        font_scale = 3.0
        thickness = 6
        
        text_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 4
        
        # 背景框
        padding = 25
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), target_color, 6)
        
        # 文字
        cv2.putText(frame, target_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, target_color, thickness)
        
        # 閃爍效果
        elapsed = time.time() - self.target_start_time
        if int(elapsed * 3) % 2 == 0:  # 每秒閃爍3次
            cv2.rectangle(frame, (bg_x1-3, bg_y1-3), (bg_x2+3, bg_y2+3), (255, 255, 255), 2)
    
    def add_result_overlay(self, frame, width, height):
        """添加結果顯示"""
        result_y = height - 180
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, result_y - 20), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 反應時間
        reaction = st.session_state.results['current_reaction']
        
        # 評級和顏色
        if reaction < 150:
            rating = "🥇 優異！"
            rating_color = (0, 255, 0)
            reaction_color = (0, 255, 0)
        elif reaction < 250:
            rating = "🥈 良好"
            rating_color = (255, 255, 0)
            reaction_color = (255, 255, 0)
        else:
            rating = "🥉 加油"
            rating_color = (255, 0, 0)
            reaction_color = (255, 100, 100)
        
        reaction_text = f"反應時間: {reaction:.0f} ms"
        cv2.putText(frame, reaction_text, 
                   (20, result_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, reaction_color, 2)
        
        # 拳速
        speed = st.session_state.results['current_speed']
        
        # 拳速評級
        if speed > 10:
            speed_rating = "💪 職業級"
            speed_color = (0, 255, 0)
        elif speed > 7:
            speed_rating = "👍 業餘級"
            speed_color = (255, 255, 0)
        elif speed > 4:
            speed_rating = "👊 健身級"
            speed_color = (255, 150, 0)
        else:
            speed_rating = "🏃 初學級"
            speed_color = (255, 100, 100)
            
        speed_text = f"出拳速度: {speed:.1f} m/s"
        cv2.putText(frame, speed_text, 
                   (20, result_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, speed_color, 2)
        
        # 評價
        cv2.putText(frame, f"評價: {rating}", 
                   (20, result_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, rating_color, 2)
        cv2.putText(frame, f"拳速: {speed_rating}", 
                   (20, result_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, speed_color, 2)

# 主應用
def main():
    st.title("🥊 拳擊反應測試系統")
    
    # 側邊欄
    with st.sidebar:
        st.header("使用說明")
        st.markdown("""
        1. **點擊『開始測試』按鈕**
        2. **集中注意力看螢幕**
        3. **看到『左拳！』或『右拳！』提示後**
        4. **快速按下對應的測試按鈕**
        5. **查看你的反應時間和拳速**
        """)
        
        st.divider()
        
        # 測試控制
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎬 開始測試", type="primary", use_container_width=True):
                if st.session_state.analyst is None:
                    st.session_state.analyst = BoxingAnalyst()
                st.session_state.analyst.start_test()
                st.session_state.test_started = True
                st.session_state.last_update = time.time()
                st.rerun()
                
        with col2:
            if st.button("🔄 重置", type="secondary", use_container_width=True):
                if st.session_state.analyst:
                    st.session_state.analyst.reset_test()
                st.session_state.results = {
                    'reaction_history': [],
                    'speed_history': [],
                    'current_reaction': 0,
                    'current_speed': 0,
                    'test_count': 0
                }
                st.session_state.test_started = False
                st.rerun()
        
        st.divider()
        
        # 手動出拳按鈕（模擬實際出拳）
        st.subheader("模擬出拳")
        st.markdown("**當看到提示時，快速點擊對應按鈕：**")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("👊 左拳", type="primary", use_container_width=True):
                if st.session_state.analyst:
                    if st.session_state.analyst.trigger_punch('LEFT'):
                        st.session_state.last_update = time.time()
                        st.rerun()
                    
        with col_right:
            if st.button("👊 右拳", type="primary", use_container_width=True):
                if st.session_state.analyst:
                    if st.session_state.analyst.trigger_punch('RIGHT'):
                        st.session_state.last_update = time.time()
                        st.rerun()
        
        st.divider()
        
        # 顯示統計數據
        st.subheader("測試統計")
        results = st.session_state.results
        
        if results['test_count'] > 0:
            st.metric("測試次數", results['test_count'])
            
            if results['reaction_history']:
                avg_reaction = np.mean(results['reaction_history'])
                best_reaction = min(results['reaction_history'])
                worst_reaction = max(results['reaction_history'])
                
                st.metric("平均反應時間", f"{avg_reaction:.0f} ms")
                st.metric("最佳反應", f"{best_reaction:.0f} ms")
                st.metric("最慢反應", f"{worst_reaction:.0f} ms")
            
            if results['speed_history']:
                avg_speed = np.mean(results['speed_history'])
                best_speed = max(results['speed_history'])
                
                st.metric("平均拳速", f"{avg_speed:.1f} m/s")
                st.metric("最快拳速", f"{best_speed:.1f} m/s")
        else:
            st.info("尚未進行測試")
        
        st.divider()
        
        st.subheader("評分標準")
        st.markdown("""
        **反應時間評級：**
        - < 150 ms: 🥇 優異 (職業級)
        - 150-250 ms: 🥈 良好 (業餘級)
        - > 250 ms: 🥉 加油 (初學級)
        
        **拳速評級：**
        - > 10 m/s: 💪 職業拳手
        - 7-10 m/s: 👍 業餘拳手
        - 4-7 m/s: 👊 健身愛好者
        - < 4 m/s: 🏃 初學者
        """)
        
        st.divider()
        
        st.info("💡 **提示**")
        st.markdown("""
        - 集中注意力看提示
        - 看到提示後立即反應
        - 保持放鬆，反應更快
        - 多練習可提升反應速度
        """)
    
    # 主內容區
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("即時分析畫面")
        
        # 創建影片顯示區域
        video_placeholder = st.empty()
        
        # 初始化分析師
        if st.session_state.analyst is None:
            st.session_state.analyst = BoxingAnalyst()
        
        analyst = st.session_state.analyst
        
        # 更新狀態（如果需要）
        if st.session_state.test_started:
            analyst.update_state()
            
            # 自動刷新畫面
            current_time = time.time()
            if current_time - st.session_state.last_update > 0.1:  # 每0.1秒更新一次
                st.session_state.last_update = current_time
                st.rerun()
        
        # 生成模擬畫面
        frame = analyst.create_simulated_frame(width=640, height=480)
        
        # 顯示畫面
        video_placeholder.image(frame, channels="BGR", width='stretch')
        
        # 控制按鈕
        col_control1, col_control2, col_control3 = st.columns(3)
        
        with col_control1:
            if st.button("⏸️ 暫停", use_container_width=True):
                st.session_state.test_started = False
                st.rerun()
                
        with col_control2:
            if st.button("▶️ 繼續", use_container_width=True):
                st.session_state.test_started = True
                st.session_state.last_update = time.time()
                st.rerun()
                
        with col_control3:
            if st.button("⏭️ 下一輪", use_container_width=True) and analyst.state == 'RESULT':
                analyst.start_test()
                st.session_state.test_started = True
                st.session_state.last_update = time.time()
                st.rerun()
        
        # 當前狀態顯示
        st.markdown("---")
        st.subheader("當前測試狀態")
        
        status_cols = st.columns(3)
        
        with status_cols[0]:
            state_text = {
                'IDLE': "🟡 待機",
                'READY': "🟢 準備",
                'COUNTDOWN': "⏱️ 倒數",
                'PUNCHING': "👊 出拳中",
                'RESULT': "📊 結果"
            }.get(analyst.state, "❓ 未知")
            st.metric("狀態", state_text)
            
        with status_cols[1]:
            if analyst.target:
                target_text = "👈 左拳" if analyst.target == 'LEFT' else "👉 右拳"
                st.metric("目標", target_text)
            else:
                st.metric("目標", "等待中")
                
        with status_cols[2]:
            if analyst.state == 'COUNTDOWN':
                remaining = max(0, analyst.countdown_end - time.time())
                st.metric("倒數", f"{remaining:.1f}s")
            elif analyst.state == 'PUNCHING':
                elapsed = time.time() - analyst.start_time
                st.metric("經過時間", f"{elapsed:.1f}s")
            else:
                st.metric("計時", "就緒")
    
    with col2:
        st.subheader("即時數據")
        
        # 當前測試數據
        if st.session_state.test_started:
            st.markdown("### 本次測試")
            
            # 反應時間
            if analyst.punch_detected:
                reaction = st.session_state.results['current_reaction']
                
                # 評級
                if reaction < 150:
                    rating = "🥇 優異"
                    delta_color = "normal"
                elif reaction < 250:
                    rating = "🥈 良好"
                    delta_color = "off"
                else:
                    rating = "🥉 加油"
                    delta_color = "inverse"
                    
                st.metric(
                    "反應時間", 
                    f"{reaction:.0f} ms",
                    delta=rating,
                    delta_color=delta_color
                )
                
                # 速度
                speed = st.session_state.results['current_speed']
                
                if speed > 10:
                    speed_rating = "💪 職業級"
                    speed_color = "normal"
                elif speed > 7:
                    speed_rating = "👍 業餘級"
                    speed_color = "off"
                elif speed > 4:
                    speed_rating = "👊 健身級"
                    speed_color = "off"
                else:
                    speed_rating = "🏃 初學級"
                    speed_color = "inverse"
                    
                st.metric(
                    "拳速",
                    f"{speed:.1f} m/s",
                    delta=speed_rating,
                    delta_color=speed_color
                )
        
        st.divider()
        
        # 速度顯示條
        st.markdown("### 拳速即時顯示")
        
        if analyst.state == 'PUNCHING' or analyst.punch_detected:
            speed = analyst.current_speed if analyst.current_speed > 0 else 0
            
            # 進度條
            progress = min(1.0, speed / 15.0)
            st.progress(progress, text=f"{speed:.1f} m/s")
            
            # 速度等級標記
            st.caption("速度參考：")
            cols_ref = st.columns(4)
            with cols_ref[0]:
                st.markdown("<small>初學 <4</small>", unsafe_allow_html=True)
            with cols_ref[1]:
                st.markdown("<small>健身 4-7</small>", unsafe_allow_html=True)
            with cols_ref[2]:
                st.markdown("<small>業餘 7-10</small>", unsafe_allow_html=True)
            with cols_ref[3]:
                st.markdown("<small>職業 >10</small>", unsafe_allow_html=True)
        
        st.divider()
        
        # 歷史數據圖表
        st.markdown("### 歷史表現趨勢")
        
        results = st.session_state.results
        
        if results['test_count'] > 0:
            import pandas as pd
            
            # 創建數據框
            test_numbers = list(range(1, results['test_count'] + 1))
            
            if len(test_numbers) == len(results['reaction_history']):
                history_data = pd.DataFrame({
                    '測試次數': test_numbers,
                    '反應時間(ms)': results['reaction_history'],
                    '拳速(m/s)': results['speed_history']
                })
                
                # 顯示最近5次
                st.dataframe(
                    history_data.tail(5),
                    width='stretch',
                    hide_index=True
                )
                
                # 簡單圖表
                if len(history_data) > 1:
                    st.line_chart(
                        history_data.set_index('測試次數'),
                        height=200
                    )
            else:
                st.info("數據同步中...")
        else:
            st.info("尚未有測試數據")

# 運行應用
if __name__ == "__main__":
    main()
