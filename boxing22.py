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
# 設定擊中距離閾值 (像素)，設為 100，只要靠近準心附近就算擊中
HIT_DISTANCE_THRESHOLD = 100 
# 估算成年人平均雙肩寬度 (米)，作為速度換算的物理參考尺
PHYSICAL_SHOULDER_WIDTH_M = 0.35

# ==========================================
# 2. 核心影像處理器 (中文提示、m/s換算、統計圖生成)
# ==========================================
class BoxingPoseProcessor:
    def __init__(self):
        # 初始化 MediaPipe Pose 模型
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_frame = None
        
        # --- 遊戲與判定狀態 ---
        self.total_hits = 0            # 當前總擊中次數
        self.state = "WAITING"         # 狀態：WAITING (準備), ACTIVE (出拳提示), FINISHED (結束)
        self.target_hand = None        # 目標手："LEFT" 或 "RIGHT"
        self.spawn_time = 0            # 提示出現時間
        self.last_reaction_time = 0.0  # 上次反應時間
        self.wait_until = time.time() + 3 # 初始準備時間 3 秒
        
        # --- 測速與統計相關變數 (單位已換算為 m/s) ---
        self.prev_time = time.time()
        self.prev_lw = None            # 上一幀左手座標 (px)
        self.prev_rw = None            # 上一幀右手座標 (px)
        self.left_speed = 0.0          # 當前左手速度 (m/s)
        self.right_speed = 0.0         # 當前右手速度 (m/s)
        self.meters_per_pixel = 0.0005 # 預設像素到米的轉換係數
        
        # --- 統計數據累加器 (單位 m/s, 秒) ---
        self.left_hits = 0
        self.right_hits = 0
        self.left_total_speed = 0.0
        self.left_total_reaction = 0.0
        self.right_total_speed = 0.0
        self.right_total_reaction = 0.0
        self.final_results = None      # 儲存最終結算結果的字典
        self.result_summary_image = None # 專門用來供使用者下載的統計結果圖片

    def generate_summary_image(self):
        # 👉 功能：整理訓練結算數據，生成一張統計圖片供下載
        if not self.final_results:
            return None
        
        r = self.final_results
        # 創建一張黑底的空白圖片 (例如 800x600)
        W, H = 800, 600
        image = Image.new("RGB", (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        t_color = (255, 255, 255) # 白色文字
        
        # 繪製標題與邊框
        draw.rectangle([(20, 20), (W-20, H-20)], outline=(0, 255, 0), width=5) # 綠色框
        draw.text((W//2 - 120, 40), "🥊 拳擊訓練成果結算 🥊", fill=(0, 255, 0))
        
        # 繪製左手統計數據區塊
        box_l = (50, 100, (W//2)-30, H-50)
        draw.rectangle(box_l, outline=(255, 100, 50), width=3) # 橘色框
        draw.text((box_l[0]+20, box_l[1]+20), f"👉 左手統計 (總數: {TOTAL_GAME_PUNCHES//2})", fill=(255, 100, 50))
        draw.text((box_l[0]+20, box_l[1]+80), f"平均拳速", fill=t_color)
        draw.text((box_l[0]+50, box_l[1]+120), f"{r['L_AVG_SPD']:.2f} m/s", fill=(255, 255, 0)) # 黃色大字
        draw.text((box_l[0]+20, box_l[1]+180), f"平均反應時間", fill=t_color)
        draw.text((box_l[0]+50, box_l[1]+220), f"{r['L_AVG_REA']:.3f} 秒", fill=(255, 255, 0))
        
        # 繪製右手統計數據區塊
        box_r = ((W//2)+30, 100, W-50, H-50)
        draw.rectangle(box_r, outline=(50, 150, 255), width=3) # 藍色框
        draw.text((box_r[0]+20, box_r[1]+20), f"👉 右手統計 (總數: {TOTAL_GAME_PUNCHES//2})", fill=(50, 150, 255))
        draw.text((box_r[0]+20, box_r[1]+80), f"平均拳速", fill=t_color)
        draw.text((box_r[0]+50, box_r[1]+120), f"{r['R_AVG_SPD']:.2f} m/s", fill=(255, 255, 0))
        draw.text((box_r[0]+20, box_r[1]+180), f"平均反應時間", fill=t_color)
        draw.text((box_r[0]+50, box_r[1]+220), f"{r['R_AVG_REA']:.3f} 秒", fill=(255, 255, 0))
        
        draw.text((W//2 - 100, H-35), f"訓練時間: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(150, 150, 150))
        
        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 取得影像並做鏡像翻轉
        img = frame.to_ndarray(format="rgb24")
        img = np.ascontiguousarray(np.fliplr(img))
        h, w, _ = img.shape
        
        if self.state == "FINISHED":
            # 👉 功能：當訓練結束時，生成統計圖片供下載，並停止畫面處理
            if self.result_summary_image is None:
                self.result_summary_image = self.generate_summary_image()
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
            
            # 取得雙肩座標以計算速度參考尺
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            # 雙肩像素距離 center = (left+right)/2
            shoulder_dist_px = np.sqrt(((ls.x - rs.x) * w)**2 + ((ls.y - rs.y) * h)**2)
            
            # 👉 核心換算：計算像素距離參考尺 (m/px)
            # 只要偵測到雙肩，就動態更新像素與物理距離的參考比例
            if shoulder_dist_px > 20: # 確保有偵測到足夠寬度
                self.meters_per_pixel = PHYSICAL_SHOULDER_WIDTH_M / shoulder_dist_px
            else:
                self.meters_per_pixel = 0.0005 # 預設參考值(~1px大約0.5mm)
            
            # 👉 C. 計算絕對準心 (下巴位置)
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
            
            # 👉 核心單位換算：計算瞬間拳速 (px/s 轉 m/s)
            if self.prev_lw and dt > 0:
                dist_l_px = np.sqrt((lw[0] - self.prev_lw[0])**2 + (lw[1] - self.prev_lw[1])**2)
                # 瞬間像素速度 (px/s)
                speed_l_px_s = dist_l_px / dt
                # 👉 將像素速度乘以參考尺，得到物理速度 (m/s)
                self.left_speed = speed_l_px_s * self.meters_per_pixel
            if self.prev_rw and dt > 0:
                dist_r_px = np.sqrt((rw[0] - self.prev_rw[0])**2 + (rw[1] - self.prev_rw[1])**2)
                speed_r_px_s = dist_r_px / dt
                self.right_speed = speed_r_px_s * self.meters_per_pixel
            self.prev_lw = lw; self.prev_rw = rw

            # 👉 D. 繪製固定目標 (下巴位置，黃色準心)
            t_color = (255, 255, 0) 
            r_circle = 15 
            draw.ellipse([target_center[0]-r_circle, target_center[1]-r_circle, target_center[0]+r_circle, target_center[1]+r_circle], outline=t_color, width=3)
            draw.line([target_center[0]-r_circle-10, target_center[1], target_center[0]+r_circle+10, target_center[1]], fill=t_color, width=2)
            draw.line([target_center[0], target_center[1]-r_circle-10, target_center[0], target_center[1]+r_circle+10], fill=t_color, width=2)

            # 👉 👉 👉 核心修正 E. 遊戲提示 (中文)，位於畫面上方 👉 👉 👉 絕對不會被擋住
            if self.state == "WAITING":
                if curr_time >= self.wait_until:
                    self.target_hand = random.choice(["LEFT", "RIGHT"])
                    self.spawn_time = curr_time
                    self.state = "ACTIVE"
                    
            elif self.state == "ACTIVE":
                # 色塊畫大一點，確保中文字清晰可見
                box_color = (255, 100, 50) if self.target_hand == "LEFT" else (50, 150, 255)
                box_l, box_t, box_r, box_b = w//2 - 120, 30, w//2 + 120, 100
                draw.rectangle([(box_l, box_t), (box_r, box_b)], fill=box_color, outline=(255, 255, 255), width=3)
                # 注意：Streamlit Cloud 的默認 PIL 字型通常不支援中文顯示 (會顯示為方塊)。
                # 我們依靠色塊分辨左右 (左橘右藍)，並嘗試寫入文字。終極解法是上傳 .ttf 字型檔並載入。
                p_text = "請出左拳！" if self.target_hand == "LEFT" else "請出右拳！"
                draw.text((box_l + 30, box_t + 25), p_text, fill=(255, 255, 255))
                
                # 👉 頭部範圍判定 (放寬閾值，150像素大約是整個頭部範圍)
                # 計算手腕與「頭部中心(鼻子)」的像素距離
                head_px_center = (int(nose.x * w), int(nose.y * h))
                relevant_wrist_px = lw if self.target_hand == "LEFT" else rw
                current_speed_m_s = self.left_speed if self.target_hand == "LEFT" else self.right_speed
                dist_to_head_px = np.sqrt((relevant_wrist_px[0] - head_px_center[0])**2 + (relevant_wrist_px[1] - head_px_center[1])**2)
                
                hit = False
                if dist_to_head_px < HIT_DISTANCE_THRESHOLD:
                    # ✅ 放寬閾值的有效出拳門檻：速度要求降低到 0.8 m/s，非常好觸發
                    if current_speed_m_s > 0.8: 
                        hit = True
                    
                # 擊中後的處理
                if hit:
                    r_time = curr_time - self.spawn_time
                    self.last_reaction_time = r_time
                    self.total_hits += 1
                    
                    if self.target_hand == "LEFT":
                        self.left_hits += 1
                        self.left_total_speed += current_speed_m_s
                        self.left_total_reaction += r_time
                    else:
                        self.right_hits += 1
                        self.right_total_speed += current_speed_m_s
                        self.right_total_reaction += r_time
                    
                    # 👉 F. 結算檢查
                    if self.total_hits >= TOTAL_GAME_PUNCHES:
                        self.state = "FINISHED"
                        # 平均拳速單位也同步為 m/s
                        self.final_results = {
                            "L_AVG_SPD": self.left_total_speed / self.left_hits if self.left_hits > 0 else 0.0,
                            "L_AVG_REA": self.left_total_reaction / self.left_hits if self.left_hits > 0 else 0.0,
                            "R_AVG_SPD": self.right_total_speed / self.right_hits if self.right_hits > 0 else 0.0,
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

        # 👉 👉 👉 核心修正 G. 儀表板數據 (中文標籤、m/s單位) 👉 👉 👉 移到畫面【下方】
        # 根據畫面高度動態決定 Y 座標，確保貼在最底下，不擋提示框
        dash_h = 130
        dash_w = 260
        margin = 15
        y1_dash = h - dash_h - margin
        y2_dash = h - margin
        draw.rectangle([(10, y1_dash), (10 + dash_w, y2_dash)], fill=(0, 0, 0), outline=(255, 255, 0), width=3) 
        
        # 儀表板數據 (中文化、單位換算為 m/s)
        draw.text((20, y1_dash + 10), "📊 即時儀表板", fill=(255, 255, 0))
        if self.state == "FINISHED":
            draw.text((20, y1_dash + 40), "訓練結束", fill=(255, 0, 0))
            draw.text((20, y1_dash + 70), f"👉 請看下方統計圖", fill=(255, 255, 255))
        else:
            dashboard_text = (
                f"訓練進度: {self.total_hits}/{TOTAL_GAME_PUNCHES}\n"
                f"上次反應: {self.last_reaction_time:.3f} 秒\n"
                f"左手爆發力: {self.left_speed:.2f} m/s\n"
                f"右手爆發力: {self.right_speed:.2f} m/s"
            )
            draw.text((20, y1_dash + 40), dashboard_text, fill=(255, 255, 255))

        # 將畫好的圖片轉回 numpy array
        img = np.array(img_pil)

        # 5. 存檔供原本畫面截圖功能使用
        self.last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# ==========================================
# 3. Streamlit UI 與主程式邏輯
# ==========================================
def main():
    st.title("🥊 拳擊反應與測速訓練器 (終極中文成果版)")
    st.markdown(f"目標固定在**下巴**位置，手腕靠近整個**頭部範圍**即算擊中。出拳提示為**中文**。完成 **{TOTAL_GAME_PUNCHES}** 拳後將自動結算結果。拳速已從像素單位 (px/s) 換算為物理參考單位 (m/s)！")

    # 顯示關於字型的提醒，避免使用者因為 PIL 顯示方塊而困惑
    st.info("💡 畫面上的中文提示框若顯示為空白方塊，是由於系統預設字型限制。我們依靠**色塊顏色**區分：**橘色代表左手**、**藍色代表右手**。成果截圖下載按鈕提供完整的訓練數據統計圖片供您保存。")

    # 啟動 WebRTC 串流
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

    # ==========================================
    # 4. 結算數據顯示區域 (當遊戲結束時自動顯示，Streamlit 原生統計組件)
    # ==========================================
    if ctx.video_processor and ctx.video_processor.state == "FINISHED":
        results = ctx.video_processor.final_results
        if results:
            st.markdown("---")
            st.header("🏆 訓練成果結算")
            st.success(f"恭喜完成 {TOTAL_GAME_PUNCHES} 拳爆發力反應訓練！這是您的統計數據 (拳速單位為 m/s)：")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👈 左手統計 (橘色)")
                # 👉 結果單位也顯示為 m/s
                st.metric(label="平均物理拳速", value=f"{results['L_AVG_SPD']:.2f} m/s")
                st.metric(label="平均反應時間", value=f"{results['L_AVG_REA']:.3f} 秒")
                
            with col2:
                st.subheader("👉 右手統計 (藍色)")
                st.metric(label="平均物理拳速", value=f"{results['R_AVG_SPD']:.2f} m/s")
                st.metric(label="平均反應時間", value=f"{results['R_AVG_REA']:.3f} 秒")
            
            # 提供一個重置按鈕 (重新整理頁面)
            if st.button("🔄 重新開始新的一局 (數據將重置)"):
                st.experimental_rerun()

    # ==========================================
    # 5. 成果截圖下載功能 (終極升級功能：下載統計圖片)
    # ==========================================
    st.markdown("---")
    st.subheader("📸 下載訓練成果圖片")
    
    if ctx.video_processor:
        if ctx.video_processor.state == "FINISHED":
            # 👉 功能：若訓練已結束，提供專門生成的【統計結果總圖】下載 👉
            if ctx.video_processor.result_summary_image is not None:
                img_to_download = ctx.video_processor.result_summary_image
                buf = io.BytesIO()
                img_to_download.save(buf, format="JPEG")
                
                st.success("統計圖片已整理完畢，請點擊下方按鈕下載完整成果統計图：")
                st.download_button(
                    label="⬇️ 下載訓練成果統計圖片 (boxing_summary.jpg)",
                    data=buf.getvalue(),
                    file_name=f"boxing_summary_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
            else:
                st.warning("正在生成統計圖片中，請稍候...")
        else:
            # ✅ 若訓練正在進行中，則保留原本的【視訊畫面截圖】功能 ✅
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
                    st.warning("請先對著鏡頭擺個姿勢，等畫面出來後再截圖喔！")
    else:
        st.info("請先點擊上方 'START' 開啟鏡頭進行訓練。")

if __name__ == "__main__":
    main()
