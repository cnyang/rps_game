#!/usr/bin/env python3
"""
剪刀石頭布遊戲 - AI 永遠贏版本
使用 MacBook 鏡頭辨識手勢，AI 故意出能贏的招

安裝：
    pip install mediapipe opencv-python numpy

執行：
    python rps_game.py
"""

import cv2
import numpy as np
import time
import os
import urllib.request

# 全域變數
mp = None
landmarker = None

def download_model():
    """下載手部偵測模型"""
    model_path = os.path.expanduser("~/hand_landmarker.task")
    if not os.path.exists(model_path):
        print("📥 正在下載手部偵測模型（約 15MB）...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("✅ 模型下載完成！")
    return model_path

def setup_mediapipe():
    """設定 MediaPipe"""
    global mp, landmarker
    
    import mediapipe as mediapipe_module
    mp = mediapipe_module
    
    model_path = download_model()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    landmarker = HandLandmarker.create_from_options(options)
    print("✅ MediaPipe 初始化成功！")
    return True


class RockPaperScissorsGame:
    def __init__(self):
        # 遊戲狀態
        self.game_state = "waiting"
        self.countdown_start = 0
        self.result_start = 0
        self.player_gesture = None
        self.ai_gesture = None
        self.score = {"ai": 0, "player": 0}
        self.frame_timestamp = 0
        
        # 贏法對照表（AI 要出什麼才能贏）
        self.winning_move = {
            "rock": "paper",      # 石頭被布包
            "paper": "scissors",  # 布被剪刀剪
            "scissors": "rock"    # 剪刀被石頭砸
        }
        
        # 顏色
        self.colors = {
            "rock": (100, 100, 255),
            "paper": (255, 200, 100),
            "scissors": (100, 255, 100),
        }
        
        # 手部連線定義
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
            (5, 9), (9, 10), (10, 11), (11, 12), # 中指
            (9, 13), (13, 14), (14, 15), (15, 16), # 無名指
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]

    def detect_gesture(self, landmarks):
        """從 landmarks 判斷手勢"""
        # landmarks: 21 個點，每個是 (x, y, z)
        
        finger_tips = [8, 12, 16, 20]   # 食指、中指、無名指、小指指尖
        finger_pips = [6, 10, 14, 18]   # 對應的第二關節
        
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            # y 值越小 = 越上面 = 手指伸直
            if landmarks[tip][1] < landmarks[pip][1]:
                fingers_extended.append(True)
            else:
                fingers_extended.append(False)
        
        index, middle, ring, pinky = fingers_extended
        count = sum(fingers_extended)
        
        # 判斷手勢
        if count <= 1:
            return "rock"       # 握拳
        elif count >= 4:
            return "paper"      # 張開
        elif index and middle and not ring and not pinky:
            return "scissors"   # 剪刀手
        elif count == 2:
            return "scissors"   # 兩指可能是剪刀
        else:
            return "paper"      # 其他當作布

    def process_frame(self, frame):
        """處理影像，偵測手勢"""
        global mp, landmarker
        
        h, w = frame.shape[:2]
        
        # 轉換為 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 偵測
        self.frame_timestamp += 33  # 約 30fps
        results = landmarker.detect_for_video(mp_image, self.frame_timestamp)
        
        gesture = None
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # 取得座標
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                
                # 繪製手部骨架
                for start, end in self.hand_connections:
                    cv2.line(frame, points[start], points[end], (0, 200, 0), 2)
                
                for point in points:
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
                
                # 辨識手勢
                gesture = self.detect_gesture(landmarks)
        
        return frame, gesture

    def get_ai_move(self, player_gesture):
        """AI 出招 - 永遠選能贏的"""
        if player_gesture is None:
            return "rock"
        return self.winning_move[player_gesture]

    def draw_gesture_box(self, frame, gesture, x, y, label):
        """繪製手勢顯示框"""
        box_w, box_h = 160, 100
        
        # 背景
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (100, 100, 100), 2)
        
        # 標籤
        cv2.putText(frame, label, (x + 10, y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 手勢
        cx, cy = x + box_w // 2, y + 60
        color = self.colors.get(gesture, (128, 128, 128))
        
        if gesture == "rock":
            cv2.circle(frame, (cx, cy), 25, color, -1)
        elif gesture == "paper":
            cv2.rectangle(frame, (cx - 25, cy - 20), (cx + 25, cy + 20), color, -1)
        elif gesture == "scissors":
            cv2.line(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), color, 4)
            cv2.line(frame, (cx + 20, cy - 20), (cx - 20, cy + 20), color, 4)
        else:
            cv2.putText(frame, "?", (cx - 12, cy + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
        
        # 手勢名稱
        name = gesture.upper() if gesture else "???"
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, name, (x + (box_w - text_size[0]) // 2, y + box_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def run(self):
        """主遊戲循環"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("\n❌ 無法開啟鏡頭！")
            print("請確認：")
            print("  1. 沒有其他程式正在使用鏡頭")
            print("  2. 已在系統設定中允許終端機存取鏡頭")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 50)
        print("🎮 剪刀石頭布 - AI 永遠贏版本")
        print("=" * 50)
        print("操作：")
        print("  SPACE = 開始")
        print("  Q     = 離開")
        print("")
        print("手勢：")
        print("  ✊ 石頭 = 握拳")
        print("  ✋ 布   = 手掌張開")
        print("  ✌️  剪刀 = 食指+中指")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 鏡像翻轉
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # 偵測手部
            frame, current_gesture = self.process_frame(frame)
            
            current_time = time.time()
            
            # ===== 遊戲狀態機 =====
            
            if self.game_state == "waiting":
                # 等待開始
                # 半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 220, h//2 - 50),
                             (w//2 + 220, h//2 + 50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                cv2.putText(frame, "Press SPACE to Start",
                           (w//2 - 175, h//2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
                
                # 顯示偵測到的手勢
                if current_gesture:
                    cv2.putText(frame, f"Detected: {current_gesture.upper()}",
                               (w//2 - 90, h//2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            elif self.game_state == "countdown":
                # 倒數計時
                elapsed = current_time - self.countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # 顯示倒數
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w//2 - 80, h//2 - 70),
                                 (w//2 + 80, h//2 + 70), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    cv2.putText(frame, str(remaining),
                               (w//2 - 25, h//2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 5)
                    
                    cv2.putText(frame, "Show your hand!",
                               (w//2 - 85, h//2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    # 倒數結束，判定結果
                    self.player_gesture = current_gesture
                    self.ai_gesture = self.get_ai_move(self.player_gesture)
                    self.game_state = "result"
                    self.result_start = current_time
                    
                    if self.player_gesture:
                        self.score["ai"] += 1
                        print(f"  你出: {self.player_gesture}, AI出: {self.ai_gesture} → AI 勝！")
                    else:
                        print("  未偵測到手勢")
            
            elif self.game_state == "result":
                # 顯示結果
                # 玩家手勢
                self.draw_gesture_box(frame, self.player_gesture, 30, 60, "YOU")
                
                # VS
                cv2.putText(frame, "VS", (130, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # AI 手勢
                self.draw_gesture_box(frame, self.ai_gesture, 200, 60, "AI")
                
                # 結果文字
                if self.player_gesture:
                    result_text = "AI WINS!"
                    color = (0, 0, 255)
                else:
                    result_text = "No hand detected"
                    color = (0, 165, 255)
                
                # 結果框
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 150, h//2 - 40),
                             (w//2 + 150, h//2 + 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                cv2.putText(frame, result_text,
                           (w//2 - text_size[0]//2, h//2 + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                
                # 3 秒後回到等待
                if current_time - self.result_start > 3:
                    self.game_state = "waiting"
            
            # ===== 固定 UI =====
            
            # 分數
            score_text = f"AI: {self.score['ai']}  |  You: {self.score['player']}"
            cv2.rectangle(frame, (0, 0), (200, 35), (0, 0, 0), -1)
            cv2.putText(frame, score_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 操作提示
            cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "SPACE: Start  |  Q: Quit",
                       (w//2 - 120, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # 顯示視窗
            cv2.imshow("Rock Paper Scissors - AI Always Wins", frame)
            
            # 鍵盤輸入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' ') and self.game_state == "waiting":
                self.game_state = "countdown"
                self.countdown_start = time.time()
                print("\n🎮 遊戲開始！")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 50)
        print("遊戲結束！")
        print(f"最終比分 - AI: {self.score['ai']}  你: {self.score['player']}")
        print("=" * 50)


def main():
    print("🎮 剪刀石頭布遊戲")
    print("正在初始化...")
    
    try:
        setup_mediapipe()
        game = RockPaperScissorsGame()
        game.run()
    except ImportError as e:
        print(f"\n❌ 缺少套件: {e}")
        print("\n請執行以下指令安裝：")
        print("  pip install mediapipe opencv-python numpy")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
