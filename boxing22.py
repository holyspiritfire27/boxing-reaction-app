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
# 設定擊中距離閾值 (像素)，數字越大越容易擊中 (閾值越低)
HIT_DISTANCE_THRESHOLD = 80 

# ==========================================
# 2. 核心影像處理器 (固定目標、距離判定、統計結算)
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
        
        # --- 測速與統計相關變數 ---
        self.prev_time = time.time()
        self.prev_lw = None            # 上一幀左手座標
        self.prev_rw = None            # 上一幀右手座標
        self.left_speed = 0.0          # 當前左手速度
        self.right_speed = 0.0         # 當前右手速度
        
        # --- 統計數據累加器 ---
        self.left_hits = 0
        self.right_hits = 0
        self.left_total_speed = 0.0
        self.left_total_reaction = 0.0
        self.right_total_speed = 0.0
        self.right_total_reaction = 0.0
        self.final_results = None      # 儲存最終結算結果的字典

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 取得影像並做鏡像翻轉
        img = frame.to_ndarray(format="rgb24")
        img = np.ascontiguousarray(np.fliplr(img))
        h, w, _ = img.shape
        
        # 如果遊戲已結束，直接回傳原始影像 (或可以加上"遊戲結束"浮水印)
        if self.state == "FINISHED":
            self.last_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        # 計算時間差
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        # 2. 進行骨架偵測
        results = self.pose.process(img)

        # 準備使用 PIL 畫圖
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 👉 A. 計算絕對準心 (嘴巴中心)
            m_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            m_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
            # 計算像素座標 center = (left+right)/2
            target_center = (int((m_left.x + m_right.x) * w / 2), 
                             int((m_left.y + m_right.y) * h / 2))
            
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

            # 👉 C. 繪製固定目標 (嘴巴位置，畫一個圓形準心)
            t_color = (255, 255, 0) # 黃色準心
            r = 20 # 準心半徑
            draw.ellipse([target_center[0]-r, target_center[1]-r, target_center[0]+r, target_center[1]+r], outline=t_color, width=3)
            # 畫十字線
            draw.line([target_center[0]-r-10, target_center[1], target_center[0]+r+10, target_center[1]], fill=t_color, width=2)
            draw.line([target_center[0], target_center[1]-r-10, target_center[0], target_center[1]+r+10], fill=t_color, width=2)

            # 👉 D. 遊戲與判定邏輯
            if self.state == "WAITING":
                if curr_time >= self.wait_until:
                    # 出拳提示出現 (隨機左或右)
                    self.target_hand = random.choice(["LEFT", "RIGHT"])
                    self.spawn_time = curr_time
                    self.state = "ACTIVE"
                    
            elif self.state == "ACTIVE":
                # 繪製中文出拳提示 (左上角)
                p_text = "請出左拳！" if self.target_hand == "LEFT" else "請出右拳！"
                p_color = (255, 100, 50) if self.target_hand == "LEFT" else (50, 150, 255)
                # 為了避免中文型設問題，我們先畫一個醒目的色塊
                draw.rectangle([(w//2 - 120, 20), (w//2 + 120, 80)], fill=p_color)
                # 注意：Streamlit Cloud 的默認字型可能不支援中文顯示，若顯示為方塊，我們用醒目色塊代替。
                # 這裡嘗試寫入文字，若失敗通常會顯示為空白或方塊。
                # 終極解法是上傳中文字型檔(.ttf)並指定載入，這裡我們先維持原狀，色塊已足夠明顯分辨左右。
                draw.text((w//2 - 100, 35), p_text, fill=(255, 255, 255))
                
                # 👉 E. 降低閾值的距離判定法
                # 計算提示手的手腕距離嘴巴中心點的距離
                relevant_wrist = lw if self.target_hand == "LEFT" else rw
                current_speed = self.left_speed if self.target_hand == "LEFT" else self.right_speed
                dist_to_target = np.sqrt((relevant_wrist[0] - target_center[0])**2 + (relevant_wrist[1] - target_center[1])**2)
                
                hit = False
                if dist_to_target < HIT_DISTANCE_THRESHOLD:
                    # 只有在速度超過一定門檻時才算"有效出拳"，避免手放在嘴邊就一直得分
                    # 這裡設定一個低門檻例如 300 px/s
                    if current_speed > 300: 
                        hit = True
                    
                # 擊中後的處理與數據累加
                if hit:
                    r_time = curr_time - self.spawn_time
                    self.last_reaction_time = r_time
                    self.total_hits += 1
                    
                    # 累加統計數據
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
                        # 計算平均值並儲存結果
                        self.final_results = {
                            "L_AVG_SPD": int(self.left_total_speed / self.left_hits) if self.left_hits > 0 else 0,
                            "L_AVG_REA": self.left_total_reaction / self.left_hits if self.left_hits > 0 else 0.0,
                            "R_AVG_SPD": int(self.right_total_speed / self.right_hits) if self.right_hits > 0 else 0,
                            "R_AVG_REA": self.right_total_reaction / self.right_hits if self.right_hits > 0 else 0.0
                        }
                    else:
                        # 隨機等待 1 ~ 2 秒後出現下一個提示
                        self.state = "WAITING"
                        self.wait_until = curr_time + random.uniform(1.0, 2.0) 

            # 3. 畫出 MediaPipe 骨架 (直接畫在 numpy array 上)
            img_with_ui = np.array(img_pil)
            mp_drawing.draw_landmarks(
                img_with_ui, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            img_pil = Image.fromarray(img_with_ui)
            draw = ImageDraw.Draw(img_pil)

        # 👉 G. 中文化儀表板 (左上角)
        dashboard_h = 110 # 增加高度以容納更多資訊
        draw.rectangle([(10, 10), (220, dashboard_h)], fill=(0, 0, 0)) 
        
        # 如果遊戲結束，顯示特殊狀態
        if self.state == "FINISHED":
            draw.text((15, 15), "🛑 訓練結束", fill=(255, 0, 0))
            draw.text((15, 45), f"請看下方統計結果", fill=(255, 255, 255))
        else:
            info_text = (
                f"進度: {self.total_hits}/{TOTAL_GAME_PUNCHES}\n"
                f"反應: {self.last_reaction_time:.3f} 秒\n"
                f"左手速: {int(self.left_speed)} px/s\n"
                f"右手速: {int(self.right_speed)} px/s"
            )
            # 注意：這裡的標籤可能因為字型問題顯示為方塊，若介意，需要上傳 .ttf 字型檔。
            draw.text((15, 15), info_text, fill=(255, 255, 255))

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
    st.markdown(f"目標固定在**嘴巴**位置，出拳提示為**中文**。完成 **{TOTAL_GAME_PUNCHES}** 拳後將自動結算結果。")

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
    # 4. 結算數據顯示區域 (當遊戲結束時自動顯示)
    # ==========================================
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
            
            # 提供一個重置按鈕 (重新整理頁面)
            if st.button("🔄 重新開始新的一局"):
                st.experimental_rerun()

    # ==========================================
    # 5. 截圖下載功能 (中文化)
    # ==========================================
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
