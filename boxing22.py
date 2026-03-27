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
st.set_page_config(page_title="Boxing Reaction App", page_icon="🥊", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. 核心影像處理器 (加入測速與反應判定)
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
        self.score = 0
        self.state = "WAITING"         # 狀態：WAITING (等標靶), ACTIVE (標靶出現)
        self.target_box = None         # 標靶座標 (x1, y1, x2, y2)
        self.target_hand = None        # 目標手："LEFT" 或 "RIGHT"
        self.spawn_time = 0            # 標靶出現時間
        self.last_reaction_time = 0.0  # 上次反應時間
        self.wait_until = time.time() + 2 # 初始給予2秒準備時間
        
        # --- 測速相關變數 ---
        self.prev_time = time.time()
        self.prev_lw = None            # 上一幀左手座標
        self.prev_rw = None            # 上一幀右手座標
        self.left_speed = 0.0          # 左手速度 (像素/秒)
        self.right_speed = 0.0         # 右手速度 (像素/秒)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. 取得影像並做鏡像翻轉 (讓動作更直覺)
        img = frame.to_ndarray(format="rgb24")
        img = np.ascontiguousarray(np.fliplr(img))
        h, w, _ = img.shape
        
        # 計算時間差 (用來算速度)
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
            
            # 取得左右手腕的座標 (將比例還原成真實像素座標)
            lw = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
            rw = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
            
            # --- 計算瞬間拳速 (距離 / 時間差) ---
            if self.prev_lw and dt > 0:
                dist_l = np.sqrt((lw[0] - self.prev_lw[0])**2 + (lw[1] - self.prev_lw[1])**2)
                self.left_speed = dist_l / dt
            if self.prev_rw and dt > 0:
                dist_r = np.sqrt((rw[0] - self.prev_rw[0])**2 + (rw[1] - self.prev_rw[1])**2)
                self.right_speed = dist_r / dt
                
            # 紀錄當前座標供下一幀使用
            self.prev_lw = lw
            self.prev_rw = rw

            # --- 遊戲邏輯：標靶生成與擊中判定 ---
            if self.state == "WAITING":
                if curr_time >= self.wait_until:
                    # 隨機決定出左手還是右手
                    box_size = 100
                    if random.choice(["LEFT", "RIGHT"]) == "LEFT":
                        self.target_hand = "LEFT"
                        # 左手標靶產生在畫面左半部
                        tx = random.randint(50, (w//2) - box_size - 10)
                    else:
                        self.target_hand = "RIGHT"
                        # 右手標靶產生在畫面右半部
                        tx = random.randint((w//2) + 10, w - box_size - 50)
                        
                    ty = random.randint(50, h//2) # 產生在畫面上半部
                    self.target_box = (tx, ty, tx+box_size, ty+box_size)
                    self.spawn_time = curr_time
                    self.state = "ACTIVE"
                    
            elif self.state == "ACTIVE" and self.target_box:
                tx1, ty1, tx2, ty2 = self.target_box
                
                # 繪製標靶 (左手用橘紅色，右手用藍色)
                target_color = (255, 100, 50) if self.target_hand == "LEFT" else (50, 150, 255)
                draw.rectangle([tx1, ty1, tx2, ty2], outline=target_color, width=6)
                draw.text((tx1+5, ty1-15), f"USE {self.target_hand} HAND!", fill=target_color)
                
                # 判定是否擊中：檢查對應的手腕座標是否進入標靶框內
                hit = False
                if self.target_hand == "LEFT" and (tx1 < lw[0] < tx2 and ty1 < lw[1] < ty2):
                    hit = True
                elif self.target_hand == "RIGHT" and (tx1 < rw[0] < tx2 and ty1 < rw[1] < ty2):
                    hit = True
                    
                # 擊中後的處理
                if hit:
                    self.last_reaction_time = curr_time - self.spawn_time
                    self.score += 1
                    self.state = "WAITING"
                    # 隨機等待 1 ~ 2.5 秒後出現下一個標靶
                    self.wait_until = curr_time + random.uniform(1.0, 2.5) 
                    self.target_box = None

            # 3. 畫出 MediaPipe 骨架 (直接畫在 numpy array 上)
            # 因為 img 已經被 PIL 轉成 img_pil，這裡先把目前的 img_pil 轉回 numpy 畫骨架
            img_with_ui = np.array(img_pil)
            mp_drawing.draw_landmarks(
                img_with_ui, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            # 把畫好骨架的圖交還給 PIL 繼續畫文字
            img_pil = Image.fromarray(img_with_ui)
            draw = ImageDraw.Draw(img_pil)

        # 4. 在左上角顯示儀表板數據
        info_text = (
            f"SCORE: {self.score}\n"
            f"REACTION: {self.last_reaction_time:.3f} sec\n"
            f"L-SPEED: {int(self.left_speed)} px/s\n"
            f"R-SPEED: {int(self.right_speed)} px/s"
        )
        # 畫一個黑色底框讓文字更清楚
        draw.rectangle([(10, 10), (180, 80)], fill=(0, 0, 0)) 
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
    st.title("🥊 拳擊反應與測速訓練 (終極版)")
    st.markdown("這個版本具備**動態標靶**、**反應時間計算**以及**左右手瞬間拳速偵測**。完全相容 Streamlit Cloud！")

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
                    label="⬇️ 下載圖片 (result.jpg)",
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
