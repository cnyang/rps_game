#!/usr/bin/env python3
"""
剪刀石頭布遊戲 - AI 永遠贏版本（可愛版）
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
import subprocess
import threading

# 全域變數
mp = None
landmarker = None


def play_system_sound(sound_name):
    """播放 macOS 系統音效（非阻塞）"""
    def _play():
        try:
            # macOS 系統音效路徑
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
    """倒數音效"""
    play_system_sound("Tink")


def play_go_sound():
    """GO 音效"""
    play_system_sound("Pop")


def play_win_sound():
    """勝利音效"""
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
        
        # 贏法對照表
        self.winning_move = {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }

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
        """處理影像，偵測手勢（不繪製骨架）"""
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
        
        return frame, gesture

    def get_ai_move(self, player_gesture):
        """AI 出招 - 永遠贏"""
        if player_gesture is None:
            return "rock"
        return self.winning_move[player_gesture]

    def draw_cute_rock(self, frame, cx, cy, size):
        """繪製可愛的石頭 ✊"""
        # 拳頭主體
        cv2.ellipse(frame, (cx, cy), (size, int(size * 1.1)), 0, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx, cy), (size, int(size * 1.1)), 0, 0, 360, (100, 120, 200), 3)
        
        # 手指關節線
        for i in range(-2, 3):
            offset = i * (size // 3)
            cv2.ellipse(frame, (cx + offset, cy - size//3), (size//5, size//6), 0, 180, 360, (150, 170, 220), 2)
        
        # 拇指
        thumb_x = cx - size - 10
        thumb_y = cy + size // 4
        cv2.ellipse(frame, (thumb_x, thumb_y), (size//3, size//2), 30, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (thumb_x, thumb_y), (size//3, size//2), 30, 0, 360, (100, 120, 200), 2)
        
        # 可愛表情
        eye_y = cy - 5
        cv2.circle(frame, (cx - 18, eye_y), 6, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 18, eye_y), 6, (50, 50, 50), -1)
        cv2.circle(frame, (cx - 18, eye_y - 2), 3, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 18, eye_y - 2), 3, (255, 255, 255), -1)
        # 微笑
        cv2.ellipse(frame, (cx, cy + 20), (15, 10), 0, 0, 180, (50, 50, 50), 2)
        # 腮紅
        cv2.circle(frame, (cx - 35, cy + 5), 8, (180, 180, 255), -1)
        cv2.circle(frame, (cx + 35, cy + 5), 8, (180, 180, 255), -1)

    def draw_cute_paper(self, frame, cx, cy, size):
        """繪製可愛的布 ✋"""
        # 手掌
        palm_h = int(size * 1.2)
        cv2.ellipse(frame, (cx, cy + size//2), (size - 10, palm_h//2), 0, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx, cy + size//2), (size - 10, palm_h//2), 0, 0, 360, (100, 120, 200), 2)
        
        # 五根手指
        finger_data = [
            (-size + 15, -size, 14, 50, -50),
            (-size//2 + 5, -size - 35, 12, 55, -15),
            (0, -size - 45, 12, 60, 0),
            (size//2 - 5, -size - 35, 12, 55, 15),
            (size - 15, -size - 15, 11, 45, 25),
        ]
        
        for dx, dy, width, height, angle in finger_data:
            fx, fy = cx + dx, cy + dy
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, (180, 200, 255), -1)
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, (100, 120, 200), 2)
        
        # 可愛表情
        eye_y = cy + size//2 - 10
        cv2.circle(frame, (cx - 18, eye_y), 6, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 18, eye_y), 6, (50, 50, 50), -1)
        cv2.circle(frame, (cx - 18, eye_y - 2), 3, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 18, eye_y - 2), 3, (255, 255, 255), -1)
        # 開心嘴巴
        cv2.ellipse(frame, (cx, cy + size//2 + 12), (12, 8), 0, 0, 180, (50, 50, 50), -1)
        # 腮紅
        cv2.circle(frame, (cx - 35, cy + size//2), 8, (180, 180, 255), -1)
        cv2.circle(frame, (cx + 35, cy + size//2), 8, (180, 180, 255), -1)

    def draw_cute_scissors(self, frame, cx, cy, size):
        """繪製可愛的剪刀 ✌️"""
        # 手掌
        palm_y = cy + size//2 + 25
        cv2.ellipse(frame, (cx, palm_y), (size - 15, size//2), 0, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx, palm_y), (size - 15, size//2), 0, 0, 360, (100, 120, 200), 2)
        
        # 食指
        pts1 = np.array([
            [cx - 8, cy + size//3],
            [cx - 35, cy - size - 15],
            [cx - 12, cy - size - 20],
            [cx + 5, cy + size//3]
        ], np.int32)
        cv2.fillPoly(frame, [pts1], (180, 200, 255))
        cv2.polylines(frame, [pts1], True, (100, 120, 200), 2)
        
        # 中指
        pts2 = np.array([
            [cx + 8, cy + size//3],
            [cx + 35, cy - size - 15],
            [cx + 12, cy - size - 20],
            [cx - 5, cy + size//3]
        ], np.int32)
        cv2.fillPoly(frame, [pts2], (180, 200, 255))
        cv2.polylines(frame, [pts2], True, (100, 120, 200), 2)
        
        # 彎曲的其他手指
        for dx in [-28, 28]:
            cv2.ellipse(frame, (cx + dx, cy + size//2 + 35), (14, 22), 0, 0, 360, (180, 200, 255), -1)
            cv2.ellipse(frame, (cx + dx, cy + size//2 + 35), (14, 22), 0, 0, 360, (100, 120, 200), 2)
        
        # 拇指
        cv2.ellipse(frame, (cx - size + 10, palm_y - 10), (16, 28), -30, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx - size + 10, palm_y - 10), (16, 28), -30, 0, 360, (100, 120, 200), 2)
        
        # 可愛表情
        cv2.circle(frame, (cx - 15, palm_y - 5), 5, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 15, palm_y - 5), 5, (50, 50, 50), -1)
        cv2.circle(frame, (cx - 15, palm_y - 7), 2, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 15, palm_y - 7), 2, (255, 255, 255), -1)
        # 眨眼笑
        cv2.ellipse(frame, (cx, palm_y + 10), (10, 6), 0, 0, 180, (50, 50, 50), 2)
        # 腮紅
        cv2.circle(frame, (cx - 32, palm_y), 7, (180, 180, 255), -1)
        cv2.circle(frame, (cx + 32, palm_y), 7, (180, 180, 255), -1)

    def draw_gesture(self, frame, gesture, cx, cy, size=70):
        """繪製手勢圖示"""
        if gesture == "rock":
            self.draw_cute_rock(frame, cx, cy, size)
        elif gesture == "paper":
            self.draw_cute_paper(frame, cx, cy, size)
        elif gesture == "scissors":
            self.draw_cute_scissors(frame, cx, cy, size)
        else:
            cv2.putText(frame, "?", (cx - 35, cy + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (150, 150, 150), 6)

    def draw_result_panel(self, frame, x, y, w, h, gesture, label, is_winner=False):
        """繪製結果面板"""
        # 背景
        if is_winner:
            cv2.rectangle(frame, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 215, 255), 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 60), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 120), 2)
        
        # 標籤
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(frame, label, (x + (w - label_size[0]) // 2, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        
        # 手勢圖示
        self.draw_gesture(frame, gesture, x + w // 2, y + h // 2 + 15, size=75)
        
        # 手勢名稱
        if gesture:
            name = gesture.upper()
            name_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, name, (x + (w - name_size[0]) // 2, y + h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

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
        print("  SPACE = 開始  |  Q = 離開")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # 偵測手部（不顯示骨架）
            frame, current_gesture = self.process_frame(frame)
            
            current_time = time.time()
            
            # ===== 遊戲狀態 =====
            
            if self.game_state == "waiting":
                # 等待畫面
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 280, h//2 - 80),
                             (w//2 + 280, h//2 + 80), (40, 40, 50), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                # 標題
                cv2.putText(frame, "ROCK  PAPER  SCISSORS",
                           (w//2 - 230, h//2 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 2)
                
                cv2.putText(frame, "Press SPACE to Start",
                           (w//2 - 160, h//2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                
                # 偵測到的手勢提示
                if current_gesture:
                    cv2.putText(frame, f"Hand detected: {current_gesture.upper()}",
                               (w//2 - 120, h//2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            elif self.game_state == "countdown":
                # 倒數計時
                elapsed = current_time - self.countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # 播放倒數音效
                    if remaining != self.last_countdown_num:
                        self.last_countdown_num = remaining
                        play_countdown_sound()
                    
                    # 倒數顯示
                    overlay = frame.copy()
                    cv2.circle(overlay, (w//2, h//2), 120, (40, 40, 50), -1)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    
                    cv2.putText(frame, str(remaining),
                               (w//2 - 40, h//2 + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)
                    
                    cv2.putText(frame, "Get Ready!",
                               (w//2 - 80, h//2 + 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                elif remaining == 0 and self.last_countdown_num != 0:
                    self.last_countdown_num = 0
                    play_go_sound()
                    self.game_state = "judging"
                    self.judge_start = current_time
            
            elif self.game_state == "judging":
                # 判定階段
                elapsed = current_time - self.judge_start
                
                # 顯示 GO!
                overlay = frame.copy()
                cv2.circle(overlay, (w//2, h//2), 120, (40, 40, 50), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                cv2.putText(frame, "GO!",
                           (w//2 - 70, h//2 + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 100), 6)
                
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
                panel_w, panel_h = 220, 250
                gap = 100
                
                # 玩家面板
                player_x = w // 2 - panel_w - gap // 2
                self.draw_result_panel(frame, player_x, h//2 - panel_h//2,
                                      panel_w, panel_h, self.player_gesture, "YOU")
                
                # VS
                cv2.putText(frame, "VS",
                           (w//2 - 30, h//2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # AI 面板
                ai_x = w // 2 + gap // 2
                self.draw_result_panel(frame, ai_x, h//2 - panel_h//2,
                                      panel_w, panel_h, self.ai_gesture, "AI", is_winner=True)
                
                # 結果文字
                if self.player_gesture:
                    result_text = "AI WINS!"
                    color = (0, 100, 255)
                else:
                    result_text = "No Hand Detected"
                    color = (0, 165, 255)
                
                text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
                cv2.putText(frame, result_text,
                           (w//2 - text_size[0]//2, h//2 - panel_h//2 - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
                
                # 3 秒後回到等待
                if current_time - self.result_start > 3:
                    self.game_state = "waiting"
                    self.last_countdown_num = 0
            
            # ===== 固定 UI =====
            
            # 分數板
            cv2.rectangle(frame, (0, 0), (300, 55), (40, 40, 50), -1)
            cv2.putText(frame, f"AI: {self.score['ai']}  |  You: {self.score['player']}",
                       (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # 底部提示
            cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 50), -1)
            cv2.putText(frame, "SPACE: Start  |  Q: Quit",
                       (w//2 - 140, h - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            # 顯示
            cv2.imshow("Rock Paper Scissors - AI Always Wins", frame)
            
            # 鍵盤
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' ') and self.game_state == "waiting":
                self.game_state = "countdown"
                self.countdown_start = time.time()
                self.last_countdown_num = 4
                print("\n🎮 遊戲開始！")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n最終比分 - AI: {self.score['ai']}  你: {self.score['player']}")


def main():
    print("🎮 剪刀石頭布遊戲（可愛版）")
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
