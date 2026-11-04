#!/usr/bin/env python3
"""
剪刀石頭布遊戲 - AI 永遠贏版本
左右分割畫面，AI 輪播動畫

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
import subprocess
import threading
import math

# 全域變數
mp = None
landmarker = None


def play_system_sound(sound_name):
    """播放 macOS 系統音效"""
    def _play():
        try:
            sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
            if os.path.exists(sound_path):
                subprocess.run(["afplay", sound_path], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        except:
            pass
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


def play_countdown_sound():
    play_system_sound("Tink")

def play_go_sound():
    play_system_sound("Pop")

def play_win_sound():
    play_system_sound("Glass")


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
        self.judge_start = 0
        self.player_gesture = None
        self.ai_gesture = None
        self.score = {"ai": 0, "player": 0}
        self.frame_timestamp = 0
        self.last_countdown_num = 0
        
        # 輪播動畫
        self.slot_index = 0
        self.slot_gestures = ["rock", "paper", "scissors"]
        self.last_slot_time = 0
        
        # 贏法對照表
        self.winning_move = {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }
        
        # 顏色
        self.ai_bg_color = (60, 60, 220)  # 紅色 (BGR)
        self.player_bg_color = (80, 80, 80)  # 灰色

    def detect_gesture(self, landmarks):
        """從 landmarks 判斷手勢"""
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                fingers_extended.append(True)
            else:
                fingers_extended.append(False)
        
        index, middle, ring, pinky = fingers_extended
        count = sum(fingers_extended)
        
        if count <= 1:
            return "rock"
        elif count >= 4:
            return "paper"
        elif index and middle and not ring and not pinky:
            return "scissors"
        elif count == 2:
            return "scissors"
        else:
            return "paper"

    def process_frame(self, frame):
        """處理影像，偵測手勢"""
        global mp, landmarker
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        self.frame_timestamp += 33
        results = landmarker.detect_for_video(mp_image, self.frame_timestamp)
        
        gesture = None
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                gesture = self.detect_gesture(landmarks)
        
        return gesture

    def get_ai_move(self, player_gesture):
        """AI 出招 - 永遠贏"""
        if player_gesture is None:
            return "rock"
        return self.winning_move[player_gesture]

    def draw_emoji_rock(self, frame, cx, cy, size):
        """繪製可愛的石頭拳頭 emoji 風格"""
        # 主體 - 黃色拳頭
        color_main = (70, 200, 255)  # 黃色 BGR
        color_dark = (30, 140, 200)  # 深黃色
        color_outline = (0, 0, 0)
        
        # 拳頭主體
        cv2.ellipse(frame, (cx, cy), (size, int(size * 1.1)), 0, 0, 360, color_main, -1)
        cv2.ellipse(frame, (cx, cy), (size, int(size * 1.1)), 0, 0, 360, color_outline, 4)
        
        # 手指關節
        for i in range(-2, 3):
            offset = i * (size // 3)
            cv2.ellipse(frame, (cx + offset, cy - size//3), 
                       (size//5, size//6), 0, 180, 360, color_dark, 3)
        
        # 拇指
        thumb_x = cx - size - 5
        thumb_y = cy + size // 3
        cv2.ellipse(frame, (thumb_x, thumb_y), (size//3, size//2), 25, 0, 360, color_main, -1)
        cv2.ellipse(frame, (thumb_x, thumb_y), (size//3, size//2), 25, 0, 360, color_outline, 3)

    def draw_emoji_paper(self, frame, cx, cy, size):
        """繪製可愛的布 emoji 風格"""
        color_main = (70, 200, 255)
        color_dark = (30, 140, 200)
        color_outline = (0, 0, 0)
        
        # 手掌
        cv2.ellipse(frame, (cx, cy + size//2), (size - 5, size//2 + 10), 0, 0, 360, color_main, -1)
        cv2.ellipse(frame, (cx, cy + size//2), (size - 5, size//2 + 10), 0, 0, 360, color_outline, 3)
        
        # 五根手指
        finger_data = [
            (-size + 20, -size + 10, 14, 45, -45),
            (-size//2 + 8, -size - 20, 13, 52, -12),
            (0, -size - 30, 13, 58, 0),
            (size//2 - 8, -size - 20, 13, 52, 12),
            (size - 15, -size + 5, 12, 42, 30),
        ]
        
        for dx, dy, width, height, angle in finger_data:
            fx, fy = cx + dx, cy + dy
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, color_main, -1)
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, color_outline, 3)

    def draw_emoji_scissors(self, frame, cx, cy, size):
        """繪製可愛的剪刀 emoji 風格"""
        color_main = (70, 200, 255)
        color_outline = (0, 0, 0)
        
        # 手掌
        palm_y = cy + size//2 + 20
        cv2.ellipse(frame, (cx, palm_y), (size - 10, size//2), 0, 0, 360, color_main, -1)
        cv2.ellipse(frame, (cx, palm_y), (size - 10, size//2), 0, 0, 360, color_outline, 3)
        
        # 食指 (V 左邊)
        pts1 = np.array([
            [cx - 5, cy + size//4],
            [cx - 30, cy - size - 10],
            [cx - 8, cy - size - 15],
            [cx + 8, cy + size//4]
        ], np.int32)
        cv2.fillPoly(frame, [pts1], color_main)
        cv2.polylines(frame, [pts1], True, color_outline, 3)
        
        # 中指 (V 右邊)
        pts2 = np.array([
            [cx + 5, cy + size//4],
            [cx + 30, cy - size - 10],
            [cx + 8, cy - size - 15],
            [cx - 8, cy + size//4]
        ], np.int32)
        cv2.fillPoly(frame, [pts2], color_main)
        cv2.polylines(frame, [pts2], True, color_outline, 3)
        
        # 彎曲的手指
        for dx in [-25, 25]:
            cv2.ellipse(frame, (cx + dx, palm_y + 5), (12, 18), 0, 0, 360, color_main, -1)
            cv2.ellipse(frame, (cx + dx, palm_y + 5), (12, 18), 0, 0, 360, color_outline, 2)
        
        # 拇指
        cv2.ellipse(frame, (cx - size + 15, palm_y - 5), (14, 24), -25, 0, 360, color_main, -1)
        cv2.ellipse(frame, (cx - size + 15, palm_y - 5), (14, 24), -25, 0, 360, color_outline, 2)

    def draw_gesture(self, frame, gesture, cx, cy, size=70):
        """繪製手勢圖示"""
        if gesture == "rock":
            self.draw_emoji_rock(frame, cx, cy, size)
        elif gesture == "paper":
            self.draw_emoji_paper(frame, cx, cy, size)
        elif gesture == "scissors":
            self.draw_emoji_scissors(frame, cx, cy, size)
        else:
            cv2.putText(frame, "?", (cx - 30, cy + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 150, 150), 5)

    def draw_small_gesture(self, frame, gesture, cx, cy, size=35):
        """繪製小型手勢圖示"""
        self.draw_gesture(frame, gesture, cx, cy, size)

    def run(self):
        """主遊戲循環"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("\n❌ 無法開啟鏡頭！")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 50)
        print("🎮 剪刀石頭布 - AI 永遠贏版本")
        print("=" * 50)
        print("  SPACE/ENTER = 開始")
        print("  Q/ESC = 離開")
        print("  ⚠️  請先點擊遊戲視窗才能使用鍵盤！")
        print("=" * 50)
        
        window_width = 1000
        window_height = 600
        half_width = window_width // 2
        
        while True:
            ret, camera_frame = cap.read()
            if not ret:
                break
            
            camera_frame = cv2.flip(camera_frame, 1)
            
            # 偵測手勢
            current_gesture = self.process_frame(camera_frame)
            
            # 創建主畫面
            frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            # ===== 左半邊：玩家 =====
            # 縮放鏡頭畫面到左半邊
            cam_h, cam_w = camera_frame.shape[:2]
            scale = min(half_width / cam_w, window_height / cam_h) * 0.85
            new_w = int(cam_w * scale)
            new_h = int(cam_h * scale)
            resized_cam = cv2.resize(camera_frame, (new_w, new_h))
            
            # 放置鏡頭畫面（置中）
            cam_x = (half_width - new_w) // 2
            cam_y = (window_height - new_h) // 2
            frame[cam_y:cam_y+new_h, cam_x:cam_x+new_w] = resized_cam
            
            # 玩家分數
            cv2.putText(frame, f"YOU: {self.score['player']}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # ===== 右半邊：AI =====
            # 紅色背景
            cv2.rectangle(frame, (half_width, 0), (window_width, window_height), 
                         self.ai_bg_color, -1)
            
            # AI 分數
            cv2.putText(frame, f"COMPUTER: {self.score['ai']}", (half_width + 20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            current_time = time.time()
            
            # ===== 遊戲狀態 =====
            
            if self.game_state == "waiting":
                # 等待畫面
                # 玩家側顯示提示
                if current_gesture:
                    # 顯示偵測到的手勢小圖示
                    self.draw_small_gesture(frame, current_gesture, 
                                           half_width // 2, window_height - 80, 40)
                
                # AI 側顯示問號
                cv2.putText(frame, "?", (half_width + half_width//2 - 50, window_height//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 8)
                
                # 中間 VS
                cv2.putText(frame, "VS", (half_width - 35, window_height // 2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 提示文字
                cv2.putText(frame, "Press SPACE to Start!", 
                           (window_width // 2 - 150, window_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            elif self.game_state == "countdown":
                # 倒數計時
                elapsed = current_time - self.countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # 播放音效
                    if remaining != self.last_countdown_num:
                        self.last_countdown_num = remaining
                        play_countdown_sound()
                    
                    # 玩家側顯示偵測到的手勢
                    if current_gesture:
                        self.draw_small_gesture(frame, current_gesture,
                                               half_width // 2, window_height - 80, 40)
                    
                    # AI 側輪播動畫
                    if current_time - self.last_slot_time > 0.1:  # 每 0.1 秒換一次
                        self.slot_index = (self.slot_index + 1) % 3
                        self.last_slot_time = current_time
                    
                    slot_gesture = self.slot_gestures[self.slot_index]
                    self.draw_gesture(frame, slot_gesture,
                                     half_width + half_width // 2, window_height // 2, 80)
                    
                    # 倒數數字（中間大）
                    cv2.putText(frame, str(remaining), 
                               (half_width - 40, window_height // 2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                
                elif remaining == 0 and self.last_countdown_num != 0:
                    self.last_countdown_num = 0
                    play_go_sound()
                    self.game_state = "judging"
                    self.judge_start = current_time
            
            elif self.game_state == "judging":
                # 判定階段 - GO!
                elapsed = current_time - self.judge_start
                
                # 繼續輪播但變慢
                if current_time - self.last_slot_time > 0.15:
                    self.slot_index = (self.slot_index + 1) % 3
                    self.last_slot_time = current_time
                
                slot_gesture = self.slot_gestures[self.slot_index]
                self.draw_gesture(frame, slot_gesture,
                                 half_width + half_width // 2, window_height // 2, 80)
                
                # GO! 文字
                cv2.putText(frame, "GO!", (half_width - 55, window_height // 2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 100), 6)
                
                # 玩家手勢
                if current_gesture:
                    self.draw_small_gesture(frame, current_gesture,
                                           half_width // 2, window_height - 80, 40)
                
                # 0.5 秒後判定
                if elapsed > 0.5:
                    self.player_gesture = current_gesture
                    self.ai_gesture = self.get_ai_move(self.player_gesture)
                    self.game_state = "result"
                    self.result_start = current_time
                    
                    if self.player_gesture:
                        self.score["ai"] += 1
                        play_win_sound()
                        print(f"  你: {self.player_gesture} vs AI: {self.ai_gesture} → AI 勝！")
                    else:
                        print("  未偵測到手勢")
            
            elif self.game_state == "result":
                # 結果畫面
                # 玩家手勢（左下角大圖）
                if self.player_gesture:
                    self.draw_gesture(frame, self.player_gesture,
                                     half_width // 2, window_height // 2 + 50, 70)
                else:
                    cv2.putText(frame, "?", (half_width // 2 - 40, window_height // 2 + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (150, 150, 150), 6)
                
                # AI 手勢（右邊大圖）
                self.draw_gesture(frame, self.ai_gesture,
                                 half_width + half_width // 2, window_height // 2, 85)
                
                # VS
                cv2.putText(frame, "VS", (half_width - 35, window_height // 2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 結果文字
                if self.player_gesture:
                    result_text = "AI WINS!"
                    # 閃爍效果
                    if int(current_time * 3) % 2 == 0:
                        cv2.putText(frame, result_text,
                                   (window_width // 2 - 100, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No Hand Detected",
                               (window_width // 2 - 140, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                
                # 3 秒後回到等待
                if current_time - self.result_start > 3:
                    self.game_state = "waiting"
                    self.last_countdown_num = 0
            
            # ===== 分隔線 =====
            cv2.line(frame, (half_width, 0), (half_width, window_height), (255, 255, 255), 3)
            
            # ===== 底部提示 =====
            cv2.rectangle(frame, (0, window_height - 35), (window_width, window_height), (30, 30, 30), -1)
            cv2.putText(frame, "SPACE: Start | Q/ESC: Quit | Click window first!",
                       (window_width // 2 - 220, window_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # 顯示
            cv2.imshow("Rock Paper Scissors - AI Always Wins", frame)
            
            # 鍵盤
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            
            if cv2.getWindowProperty("Rock Paper Scissors - AI Always Wins", cv2.WND_PROP_VISIBLE) < 1:
                break
            
            if (key == ord(' ') or key == 13) and self.game_state == "waiting":
                self.game_state = "countdown"
                self.countdown_start = time.time()
                self.last_countdown_num = 4
                self.last_slot_time = time.time()
                print("\n🎮 遊戲開始！")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n最終比分 - AI: {self.score['ai']}  你: {self.score['player']}")


def main():
    print("🎮 剪刀石頭布遊戲")
    print("正在初始化...\n")
    
    try:
        setup_mediapipe()
        game = RockPaperScissorsGame()
        game.run()
    except ImportError as e:
        print(f"\n❌ 缺少套件: {e}")
        print("\n請執行：")
        print("  pip install mediapipe opencv-python numpy")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
