import streamlit as st
import av
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import io
import time
import random
from streamlit_webrtc import webrtc_streamer

# ==========================================
# 1. 頁面與 MediaPipe 初始化
# ==========================================
st.set_page_config(page_title="拳擊反應訓練器", page_icon="🥊", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 設定遊戲總拳數
TOTAL_GAME_PUNCHES = 10
# 設定擊中距離閾值 (像素)：設為 150，只要靠近整個頭部範圍就算擊中！
HIT_DISTANCE_THRESHOLD = 150 

# ==========================================
# 2. 核心影像處理器 (下巴目標、頭部範圍判定、UI調整)
# ==========================================
class BoxingPoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_frame = None
        
        # --- 遊戲與判定狀態 ---
        self.total_hits = 0            
        self.state = "WAITING"         
        self.target_hand = None        
        self.spawn_time = 0            
        self.last_reaction_time = 0.0  
        self.wait_until = time.time() + 3 
        
        # --- 測速與統計相關變數 ---
        self.prev_time = time.time()
        self.prev_lw = None            
        self.prev_rw = None            
        self.left_speed = 0.0          
        self.right_speed = 0.0         
        
        # --- 統計數據累加器 ---
        self.left_hits = 0
        self.right_hits = 0
        self.left_total_speed = 0.0
        self.left_total_reaction = 0.0
        self.right_total_speed = 0.0
        self.right_total_reaction = 0.0
        self.final_results = None      

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 取得影像並做鏡像翻轉
        img = frame.to_ndarray(format="rgb24")
        img = np.ascontiguousarray(np.fliplr(img))
        h, w, _ = img.shape
        
        if self.state == "FINISHED":
            self.last_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        # 計算時間差
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        # 2. 進行骨架偵測
        results = self.pose.process(img)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 👉 A. 計算絕對準心 (推算下巴位置)
            m_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            m_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            
            # 取得鼻子和嘴巴的真實像素位置
            nose_y_px = nose.y * h
            mouth_y_px = (m_left.y + m_right.y) / 2 * h
            mouth_x_px = (m_left.x + m_right.x) / 2 * w
            
            # 下巴大約在嘴巴正下方，距離約等於「鼻子到嘴巴的距離的 1.2 倍」
            chin_offset = (mouth_y_px - nose_y_px) * 1.2
            target_center = (int(mouth_x_px), int(mouth_y_px + chin_offset))
            
            # 頭部中心(以鼻子為基準)，用來做寬鬆的擊中判定
            head_center = (int(nose.x * w), int(nose.y * h))
            
            # 取得左右手腕的座標
            lw = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
            rw = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
            
            # 👉 B. 計算瞬間拳速 (px/s)
            if self.prev_lw and dt > 0:
                dist_l = np.sqrt((lw[0] - self.prev_lw[0])**2 + (lw[1] - self.prev_lw[1])**2)
                self.left_speed = dist_l / dt
            if self.prev_rw and dt > 0:
                dist_r = np.sqrt((rw[0] - self.prev_rw[0])**2 + (rw[1] - self.prev_rw[1])**2)
                self.right_speed = dist_r / dt
            self.prev_lw = lw; self.prev_rw = rw

            # 👉 C. 繪製固定目標 (下巴位置，黃色準心)
            t_color = (255, 255, 0) 
            r = 15 
            draw.ellipse([target_center[0]-r, target_center[1]-r, target_center[0]+r, target_center[1]+r], outline=t_color, width=3)
            draw.line([target_center[0]-r-10, target_center[1], target_center[0]+r+10, target_center[1]], fill=t_color, width=2)
            draw.line([target_center[0], target_center[1]-r-10, target_center[0], target_center[1]+r+10], fill=t_color, width=2)

            # 👉 D. 遊戲與判定邏輯
            if self.state == "WAITING":
                if curr_time >= self.wait_until:
                    self.target_hand = random.choice(["LEFT", "RIGHT"])
                    self.spawn_time = curr_time
                    self.state = "ACTIVE"
                    
            elif self.state == "ACTIVE":
                # 繪製中文出拳提示 (正上方，絕對不會被擋住)
                p_text = "請出左拳！" if self.target_hand == "LEFT" else "請出右拳！"
                p_color = (255, 100, 50) if self.target_hand == "LEFT" else (50, 150, 255)
                # 色塊稍微畫大一點，確保中文字能塞進去
                draw.rectangle([(w//2 - 120, 20), (w//2 + 120, 80)], fill=p_color)
                draw.text((w//2 - 50, 40), p_text, fill=(255, 255, 255))
                
                # 👉 E. 放寬閾值的頭部範圍判定
                # 計算手腕與「頭部中心(鼻子)」的距離，而不是小小的下巴
                relevant_wrist = lw if self.target_hand == "LEFT" else rw
                current_speed = self.left_speed if self.target_hand == "LEFT" else self.right_speed
                dist_to_head = np.sqrt((relevant_wrist[0] - head_center[0])**2 + (relevant_wrist[1] - head_center[1])**2)
                
                hit = False
                # 只要距離小於 150 (大約涵蓋整個臉部範圍)
                if dist_to_head < HIT_DISTANCE_THRESHOLD:
                    # 最低速度要求降低到 200，更好觸發
                    if current_speed > 200: 
                        hit = True
                    
                # 擊中後的處理
                if hit:
                    r_time = curr_time - self.spawn_time
                    self.last_reaction_time = r_time
                    self.total_hits += 1
                    
                    if self.target_hand == "LEFT":
                        self.left_hits += 1
                        self.left_total_speed += current_speed
                        self.left_total_reaction += r_time
                    else:
                        self.right_hits += 1
                        self.right_total_speed += current_speed
                        self.right_total_reaction += r_time
                    
                    # 👉 F. 結算檢查
                    if self.total_hits >= TOTAL_GAME_PUNCHES:
                        self.state = "FINISHED"
                        self.final_results = {
                            "L_AVG_SPD": int(self.left_total_speed / self.left_hits) if self.left_hits > 0 else 0,
                            "L_AVG_REA": self.left_total_reaction / self.left_hits if self.left_hits > 0 else 0.0,
                            "R_AVG_SPD": int(self.right_total_speed / self.right_hits) if self.right_hits > 0 else 0,
                            "R_AVG_REA": self.right_total_reaction / self.right_hits if self.right_hits > 0 else 0.0
                        }
                    else:
                        self.state = "WAITING"
                        self.wait_until = curr_time + random.uniform(1.0, 2.0) 

            # 3. 畫出 MediaPipe 骨架
            img_with_ui = np.array(img_pil)
            mp_drawing.draw_landmarks(
                img_with_ui, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            img_pil = Image.fromarray(img_with_ui)
            draw = ImageDraw.Draw(img_pil)

        # 👉 G. 儀表板移到畫面【左下角】
        box_width = 220
        box_height = 110
        # 根據畫面高度動態決定 Y 座標，確保貼在最底下
        margin = 15
        y1 = h - box_height - margin
        y2 = h - margin
        draw.rectangle([(10, y1), (10 + box_width, y2)], fill=(0, 0, 0)) 
        
        if self.state == "FINISHED":
            draw.text((15, y1 + 10), "🛑 訓練結束", fill=(255, 0, 0))
            draw.text((15, y1 + 40), f"請看下方統計結果", fill=(255, 255, 255))
        else:
            info_text = (
                f"進度: {self.total_hits}/{TOTAL_GAME_PUNCHES}\n"
                f"反應: {self.last_reaction_time:.3f} 秒\n"
                f"左手速: {int(self.left_speed)} px/s\n"
                f"右手速: {int(self.right_speed)} px/s"
            )
            draw.text((15, y1 + 10), info_text, fill=(255, 255, 255))

        # 將畫好的圖片轉回 numpy array
        img = np.array(img_pil)

        # 5. 存檔供截圖，並回傳
        self.last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# ==========================================
# 3. Streamlit UI 與主程式邏輯
# ==========================================
def main():
    st.title("🥊 拳擊反應與測速訓練器 (中文結算版)")
    st.markdown(f"目標固定在**下巴**位置，出拳提示為**中文**。完成 **{TOTAL_GAME_PUNCHES}** 拳後將自動結算結果。")

    ctx = webrtc_streamer(
        key="boxing-reaction",
        video_processor_factory=BoxingPoseProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor and ctx.video_processor.state == "FINISHED":
        results = ctx.video_processor.final_results
        if results:
            st.markdown("---")
            st.header("🏆 訓練成果結算")
            st.success(f"恭喜完成 {TOTAL_GAME_PUNCHES} 拳反應訓練！這是你的統計數據：")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👈 左手統計")
                st.metric(label="平均拳速", value=f"{results['L_AVG_SPD']} px/s")
                st.metric(label="平均反應時間", value=f"{results['L_AVG_REA']:.3f} 秒")
                
            with col2:
                st.subheader("👉 右手統計")
                st.metric(label="平均拳速", value=f"{results['R_AVG_SPD']} px/s")
                st.metric(label="平均反應時間", value=f"{results['R_AVG_REA']:.3f} 秒")
            
            if st.button("🔄 重新開始新的一局"):
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("📸 成果截圖")
    
    if ctx.video_processor:
        if st.button("擷取當下畫面"):
            if ctx.video_processor.last_frame is not None:
                img_to_save = ctx.video_processor.last_frame
                image = Image.fromarray(img_to_save)
                
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                
                st.success("截圖成功！請點擊下方按鈕下載：")
                st.download_button(
                    label="⬇️ 下載圖片 (boxing_result.jpg)",
                    data=buf.getvalue(),
                    file_name="boxing_result.jpg",
                    mime="image/jpeg"
                )
            else:
                st.warning("請先對著鏡頭擺個姿勢，等畫面出來後再截圖喔！")
    else:
        st.info("請先點擊上方 'START' 開啟鏡頭。")

if __name__ == "__main__":
    main()
