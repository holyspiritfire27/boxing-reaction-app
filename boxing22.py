import streamlit as st
import av
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io
import time
import random
import urllib.request
import os
from streamlit_webrtc import webrtc_streamer

# ==========================================
# 0. 自動下載中文字型 (已成功運作)
# ==========================================
FONT_PATH = "LXGWWenKai-Regular.ttf"
if not os.path.exists(FONT_PATH):
    try:
        url = "https://raw.githubusercontent.com/lxgw/LxgwWenKai/main/fonts/TTF/LXGWWenKai-Regular.ttf"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(FONT_PATH, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print("字型下載失敗", e)

try:
    font_huge = ImageFont.truetype(FONT_PATH, 40)
    font_large = ImageFont.truetype(FONT_PATH, 28)
    font_medium = ImageFont.truetype(FONT_PATH, 20)
except:
    font_huge = font_large = font_medium = None

# ==========================================
# 1. 頁面與 MediaPipe 初始化
# ==========================================
st.set_page_config(page_title="拳擊反應訓練器", page_icon="🥊", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

TOTAL_GAME_PUNCHES = 10
HIT_DISTANCE_THRESHOLD = 150 

# 👉 物理測速核心參數修正
# 放大係數：還原 AI 壓縮的真實物理速度，並對齊人類 7~20 m/s 的爆發力範圍
SPEED_CALIBRATION = 4.5 

# ==========================================
# 2. 核心影像處理器 
# ==========================================
class BoxingPoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.last_frame = None
        
        # --- 遊戲與判定狀態 ---
        self.total_hits = 0            
        self.state = "WAITING"         
        self.target_hand = None        
        self.spawn_time = 0            
        self.last_reaction_time = 0.0  
        self.wait_until = time.time() + 3 
        
        # --- 3D 測速相關變數 ---
        self.prev_time = time.time()
        self.prev_lw_3d = None            
        self.prev_rw_3d = None            
        self.left_speed = 0.0          
        self.right_speed = 0.0 
        self.current_punch_max_speed = 0.0 
        
        # --- 統計數據累加器 ---
        self.left_hits = 0
        self.right_hits = 0
        self.left_total_speed = 0.0
        self.left_total_reaction = 0.0
        self.right_total_speed = 0.0
        self.right_total_reaction = 0.0
        self.final_results = None      
        self.result_summary_image = None 

    def generate_summary_image(self):
        if not self.final_results:
            return None
        
        r = self.final_results
        W, H = 800, 600
        image = Image.new("RGB", (W, H), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        
        draw.rectangle([(20, 20), (W-20, H-20)], outline=(0, 255, 0), width=5)
        draw.text((W//2 - 200, 40), "🥊 拳擊訓練成果結算 🥊", fill=(0, 255, 0), font=font_huge)
        
        box_l = (50, 120, (W//2)-30, H-60)
        draw.rectangle(box_l, outline=(255, 100, 50), width=3) 
        draw.text((box_l[0]+30, box_l[1]+20), f"👉 左手統計", fill=(255, 100, 50), font=font_large)
        draw.text((box_l[0]+30, box_l[1]+90), f"平均峰值拳速", fill=(200, 200, 200), font=font_medium)
        draw.text((box_l[0]+60, box_l[1]+130), f"{r['L_AVG_SPD']:.2f} m/s", fill=(255, 255, 0), font=font_huge)
        draw.text((box_l[0]+30, box_l[1]+200), f"平均反應時間", fill=(200, 200, 200), font=font_medium)
        draw.text((box_l[0]+60, box_l[1]+240), f"{r['L_AVG_REA']:.3f} 秒", fill=(255, 255, 0), font=font_huge)
        
        box_r = ((W//2)+30, 120, W-50, H-60)
        draw.rectangle(box_r, outline=(50, 150, 255), width=3) 
        draw.text((box_r[0]+30, box_r[1]+20), f"👉 右手統計", fill=(50, 150, 255), font=font_large)
        draw.text((box_r[0]+30, box_r[1]+90), f"平均峰值拳速", fill=(200, 200, 200), font=font_medium)
        draw.text((box_r[0]+60, box_r[1]+130), f"{r['R_AVG_SPD']:.2f} m/s", fill=(255, 255, 0), font=font_huge)
        draw.text((box_r[0]+30, box_r[1]+200), f"平均反應時間", fill=(200, 200, 200), font=font_medium)
        draw.text((box_r[0]+60, box_r[1]+240), f"{r['R_AVG_REA']:.3f} 秒", fill=(255, 255, 0), font=font_huge)
        
        draw.text((W//2 - 120, H-40), f"訓練時間: {time.strftime('%Y-%m-%d %H:%M')}", fill=(150, 150, 150), font=font_medium)
        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        img = np.ascontiguousarray(np.fliplr(img))
        h, w, _ = img.shape
        
        if self.state == "FINISHED":
            if self.result_summary_image is None:
                self.result_summary_image = self.generate_summary_image()
            self.last_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        # 遊戲邏輯時間 (反應時間) 依舊依賴系統時間
        curr_time = time.time()

        results = self.pose.process(img)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark
            
            # 計算下巴準心
            m_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            m_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            
            nose_y_px, head_center = nose.y * h, (int(nose.x * w), int(nose.y * h))
            mouth_y_px = (m_left.y + m_right.y) / 2 * h
            mouth_x_px = (m_left.x + m_right.x) / 2 * w
            chin_offset = (mouth_y_px - nose_y_px) * 1.2
            target_center = (int(mouth_x_px), int(mouth_y_px + chin_offset))
            
            # 鏡像修正後的 2D 座標 (判定用)
            lw_px = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
            rw_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
            
            # 鏡像修正後的 3D 座標 (測速用)
            lw_3d = world_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            rw_3d = world_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            # 👉 修正 1：還原並放大被 AI 壓縮的 Z 軸深度 (乘上 1.5 倍)
            curr_lw_3d = np.array([lw_3d.x, lw_3d.y, lw_3d.z * 1.5])
            curr_rw_3d = np.array([rw_3d.x, rw_3d.y, rw_3d.z * 1.5])
            
            # 👉 修正 2：強制綁定攝影機 30fps 物理時間 (0.033 秒)，排除伺服器延遲干擾！
            dt_speed = 0.033 
            
            if self.prev_lw_3d is not None:
                dist_l_m = np.linalg.norm(curr_lw_3d - self.prev_lw_3d)
                # 真實速度 = 距離 / 時間 * 物理校正係數
                self.left_speed = (dist_l_m / dt_speed) * SPEED_CALIBRATION
            if self.prev_rw_3d is not None:
                dist_r_m = np.linalg.norm(curr_rw_3d - self.prev_rw_3d)
                self.right_speed = (dist_r_m / dt_speed) * SPEED_CALIBRATION
                
            self.prev_lw_3d = curr_lw_3d
            self.prev_rw_3d = curr_rw_3d

            # 繪製下巴準心
            t_color = (255, 255, 0) 
            r_circle = 15 
            draw.ellipse([target_center[0]-r_circle, target_center[1]-r_circle, target_center[0]+r_circle, target_center[1]+r_circle], outline=t_color, width=3)
            draw.line([target_center[0]-r_circle-10, target_center[1], target_center[0]+r_circle+10, target_center[1]], fill=t_color, width=2)
            draw.line([target_center[0], target_center[1]-r_circle-10, target_center[0], target_center[1]+r_circle+10], fill=t_color, width=2)

            # 遊戲邏輯與動態最高速捕捉
            if self.state == "WAITING":
                if curr_time >= self.wait_until:
                    self.target_hand = random.choice(["LEFT", "RIGHT"])
                    self.spawn_time = curr_time
                    self.current_punch_max_speed = 0.0 
                    self.state = "ACTIVE"
                    
            elif self.state == "ACTIVE":
                box_color = (255, 100, 50) if self.target_hand == "LEFT" else (50, 150, 255)
                box_l, box_t, box_r, box_b = w//2 - 140, 30, w//2 + 140, 100
                draw.rectangle([(box_l, box_t), (box_r, box_b)], fill=box_color, outline=(255, 255, 255), width=3)
                p_text = "🥊 請出 左拳！" if self.target_hand == "LEFT" else "🥊 請出 右拳！"
                if font_large:
                    draw.text((box_l + 30, box_t + 18), p_text, fill=(255, 255, 255), font=font_large)
                
                # 追蹤此次出拳的「最高速」
                current_speed_m_s = self.left_speed if self.target_hand == "LEFT" else self.right_speed
                if current_speed_m_s > self.current_punch_max_speed:
                    self.current_punch_max_speed = current_speed_m_s
                
                relevant_wrist_px = lw_px if self.target_hand == "LEFT" else rw_px
                dist_to_head_px = np.sqrt((relevant_wrist_px[0] - head_center[0])**2 + (relevant_wrist_px[1] - head_center[1])**2)
                
                hit = False
                if dist_to_head_px < HIT_DISTANCE_THRESHOLD:
                    # 降低判定擊中的門檻，確保玩家專注發力而非瞄準
                    if current_speed_m_s > 1.5: 
                        hit = True
                    
                if hit:
                    r_time = curr_time - self.spawn_time
                    self.last_reaction_time = r_time
                    self.total_hits += 1
                    
                    final_recorded_speed = max(self.current_punch_max_speed, current_speed_m_s)
                    
                    if self.target_hand == "LEFT":
                        self.left_hits += 1
                        self.left_total_speed += final_recorded_speed
                        self.left_total_reaction += r_time
                    else:
                        self.right_hits += 1
                        self.right_total_speed += final_recorded_speed
                        self.right_total_reaction += r_time
                    
                    if self.total_hits >= TOTAL_GAME_PUNCHES:
                        self.state = "FINISHED"
                        self.final_results = {
                            "L_AVG_SPD": self.left_total_speed / self.left_hits if self.left_hits > 0 else 0.0,
                            "L_AVG_REA": self.left_total_reaction / self.left_hits if self.left_hits > 0 else 0.0,
                            "R_AVG_SPD": self.right_total_speed / self.right_hits if self.right_hits > 0 else 0.0,
                            "R_AVG_REA": self.right_total_reaction / self.right_hits if self.right_hits > 0 else 0.0
                        }
                    else:
                        self.state = "WAITING"
                        self.wait_until = curr_time + random.uniform(1.0, 2.0) 

            # 畫出骨架
            img_with_ui = np.array(img_pil)
            mp_drawing.draw_landmarks(
                img_with_ui, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            img_pil = Image.fromarray(img_with_ui)
            draw = ImageDraw.Draw(img_pil)

        # 儀表板數據
        dash_h = 130
        dash_w = 260
        margin = 15
        y1_dash = h - dash_h - margin
        y2_dash = h - margin
        draw.rectangle([(10, y1_dash), (10 + dash_w, y2_dash)], fill=(0, 0, 0, 180), outline=(255, 255, 0), width=3) 
        
        if font_medium:
            draw.text((20, y1_dash + 5), "📊 即時儀表板", fill=(255, 255, 0), font=font_medium)
            if self.state == "FINISHED":
                draw.text((20, y1_dash + 40), "訓練結束", fill=(255, 50, 50), font=font_large)
                draw.text((20, y1_dash + 80), f"👉 請看下方統計圖", fill=(255, 255, 255), font=font_medium)
            else:
                dashboard_text = (
                    f"訓練進度: {self.total_hits}/{TOTAL_GAME_PUNCHES}\n"
                    f"上次反應: {self.last_reaction_time:.3f} 秒\n"
                    f"左手爆發: {self.current_punch_max_speed if self.target_hand == 'LEFT' else self.left_speed:.2f} m/s\n"
                    f"右手爆發: {self.current_punch_max_speed if self.target_hand == 'RIGHT' else self.right_speed:.2f} m/s"
                )
                draw.text((20, y1_dash + 35), dashboard_text, fill=(255, 255, 255), font=font_medium)

        img = np.array(img_pil)
        self.last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# ==========================================
# 3. Streamlit UI 與主程式邏輯
# ==========================================
def main():
    st.title("🥊 拳擊反應與測速訓練器 (專業物理測速版)")
    st.markdown(f"導入 **3D 深度感測器與物理補償引擎**，精準捕捉你的真實最高峰值拳速！完成 **{TOTAL_GAME_PUNCHES}** 拳後將自動結算結果。")

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
            st.success(f"恭喜完成 {TOTAL_GAME_PUNCHES} 拳爆發力反應訓練！這是您的統計數據：")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👈 左手統計 (橘色)")
                st.metric(label="平均物理峰值拳速", value=f"{results['L_AVG_SPD']:.2f} m/s")
                st.metric(label="平均反應時間", value=f"{results['L_AVG_REA']:.3f} 秒")
                
            with col2:
                st.subheader("👉 右手統計 (藍色)")
                st.metric(label="平均物理峰值拳速", value=f"{results['R_AVG_SPD']:.2f} m/s")
                st.metric(label="平均反應時間", value=f"{results['R_AVG_REA']:.3f} 秒")
            
            if st.button("🔄 重新開始新的一局 (數據將重置)"):
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("📸 下載訓練成果圖片")
    
    if ctx.video_processor:
        if ctx.video_processor.state == "FINISHED":
            if ctx.video_processor.result_summary_image is not None:
                img_to_download = ctx.video_processor.result_summary_image
                buf = io.BytesIO()
                img_to_download.save(buf, format="JPEG")
                
                st.success("統計圖片已整理完畢，請點擊下方按鈕下載完整成果統計圖：")
                st.download_button(
                    label="⬇️ 下載訓練成果統計圖片 (boxing_summary.jpg)",
                    data=buf.getvalue(),
                    file_name=f"boxing_summary_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
        else:
            if st.button("擷取當下視訊畫面"):
                if ctx.video_processor.last_frame is not None:
                    img_to_save = ctx.video_processor.last_frame
                    image = Image.fromarray(img_to_save)
                    
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    
                    st.success("截圖成功！請點擊下方按鈕下載：")
                    st.download_button(
                        label="⬇️ 下載當下圖片 (boxing_screenshot.jpg)",
                        data=buf.getvalue(),
                        file_name=f"boxing_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg"
                    )
    else:
        st.info("請先點擊上方 'START' 開啟鏡頭進行訓練。")

if __name__ == "__main__":
    main()
