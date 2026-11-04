#!/usr/bin/env python3
"""
剪刀石頭布遊戲 - AI 永遠贏版本（可愛版）
使用 MacBook 鏡頭辨識手勢，AI 故意出能贏的招

安裝：
    pip install mediapipe opencv-python numpy pygame

執行：
    python rps_game.py
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import threading

# 音效模組
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    SOUND_ENABLED = True
except:
    SOUND_ENABLED = False
    print("⚠️  pygame 未安裝，音效將被停用")
    print("   安裝音效支援: pip install pygame")

# 全域變數
mp = None
landmarker = None


def generate_beep(frequency=440, duration=0.1, volume=0.3):
    """生成 beep 音效"""
    if not SOUND_ENABLED:
        return None
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    # 加入淡出效果
    fade = np.linspace(1, 0, n_samples // 4)
    wave[-len(fade):] *= fade
    # 轉換為 stereo
    stereo = np.column_stack((wave, wave))
    sound_array = (stereo * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)


def generate_win_sound():
    """生成勝利音效（上升音階）"""
    if not SOUND_ENABLED:
        return None
    sample_rate = 44100
    duration = 0.4
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    # 上升頻率
    freq = np.linspace(400, 800, n_samples)
    wave = np.sin(2 * np.pi * freq * t / sample_rate * np.arange(n_samples)) * 0.3
    fade = np.linspace(1, 0, n_samples // 3)
    wave[-len(fade):] *= fade
    stereo = np.column_stack((wave, wave))
    sound_array = (stereo * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)


def generate_countdown_sound():
    """生成倒數音效"""
    if not SOUND_ENABLED:
        return None
    return generate_beep(600, 0.15, 0.4)


def generate_go_sound():
    """生成 GO 音效"""
    if not SOUND_ENABLED:
        return None
    return generate_beep(800, 0.3, 0.5)


# 預先生成音效
SOUNDS = {}
if SOUND_ENABLED:
    SOUNDS['countdown'] = generate_countdown_sound()
    SOUNDS['go'] = generate_go_sound()
    SOUNDS['win'] = generate_win_sound()


def play_sound(name):
    """播放音效（非阻塞）"""
    if SOUND_ENABLED and name in SOUNDS and SOUNDS[name]:
        SOUNDS[name].play()


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
        self.last_countdown_num = 0
        
        # 贏法對照表
        self.winning_move = {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }
        
        # 手部連線
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

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
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                
                # 繪製手部
                for start, end in self.hand_connections:
                    cv2.line(frame, points[start], points[end], (0, 220, 0), 3)
                for point in points:
                    cv2.circle(frame, point, 6, (0, 255, 100), -1)
                    cv2.circle(frame, point, 6, (0, 200, 0), 2)
                
                gesture = self.detect_gesture(landmarks)
        
        return frame, gesture

    def get_ai_move(self, player_gesture):
        """AI 出招 - 永遠贏"""
        if player_gesture is None:
            return "rock"
        return self.winning_move[player_gesture]

    def draw_cute_rock(self, frame, cx, cy, size):
        """繪製可愛的石頭 ✊"""
        # 拳頭主體（膚色）
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
        cv2.circle(frame, (cx - 15, eye_y), 5, (50, 50, 50), -1)  # 左眼
        cv2.circle(frame, (cx + 15, eye_y), 5, (50, 50, 50), -1)  # 右眼
        cv2.circle(frame, (cx - 15, eye_y - 2), 2, (255, 255, 255), -1)  # 眼睛亮點
        cv2.circle(frame, (cx + 15, eye_y - 2), 2, (255, 255, 255), -1)
        # 微笑
        cv2.ellipse(frame, (cx, cy + 15), (12, 8), 0, 0, 180, (50, 50, 50), 2)

    def draw_cute_paper(self, frame, cx, cy, size):
        """繪製可愛的布 ✋"""
        # 手掌
        palm_h = int(size * 1.2)
        cv2.ellipse(frame, (cx, cy + size//2), (size - 10, palm_h//2), 0, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx, cy + size//2), (size - 10, palm_h//2), 0, 0, 360, (100, 120, 200), 2)
        
        # 五根手指
        finger_data = [
            (-size + 15, -size, 12, 45, -50),   # 拇指
            (-size//2 + 5, -size - 30, 10, 50, -15),   # 食指
            (0, -size - 40, 10, 55, 0),          # 中指
            (size//2 - 5, -size - 30, 10, 50, 15),    # 無名指
            (size - 15, -size - 10, 9, 40, 25),       # 小指
        ]
        
        for dx, dy, width, height, angle in finger_data:
            fx, fy = cx + dx, cy + dy
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, (180, 200, 255), -1)
            cv2.ellipse(frame, (fx, fy), (width, height), angle, 0, 360, (100, 120, 200), 2)
        
        # 可愛表情
        eye_y = cy + size//2 - 10
        cv2.circle(frame, (cx - 15, eye_y), 5, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 15, eye_y), 5, (50, 50, 50), -1)
        cv2.circle(frame, (cx - 15, eye_y - 2), 2, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 15, eye_y - 2), 2, (255, 255, 255), -1)
        # 開心的嘴巴
        cv2.ellipse(frame, (cx, cy + size//2 + 10), (10, 6), 0, 0, 180, (50, 50, 50), -1)

    def draw_cute_scissors(self, frame, cx, cy, size):
        """繪製可愛的剪刀 ✌️"""
        # 手掌（較小，在下方）
        palm_y = cy + size//2 + 20
        cv2.ellipse(frame, (cx, palm_y), (size - 20, size//2), 0, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx, palm_y), (size - 20, size//2), 0, 0, 360, (100, 120, 200), 2)
        
        # 食指（V 的左邊）
        pts1 = np.array([
            [cx - 5, cy + size//3],
            [cx - 30, cy - size - 10],
            [cx - 10, cy - size - 15],
            [cx + 5, cy + size//3]
        ], np.int32)
        cv2.fillPoly(frame, [pts1], (180, 200, 255))
        cv2.polylines(frame, [pts1], True, (100, 120, 200), 2)
        
        # 中指（V 的右邊）
        pts2 = np.array([
            [cx + 5, cy + size//3],
            [cx + 30, cy - size - 10],
            [cx + 10, cy - size - 15],
            [cx - 5, cy + size//3]
        ], np.int32)
        cv2.fillPoly(frame, [pts2], (180, 200, 255))
        cv2.polylines(frame, [pts2], True, (100, 120, 200), 2)
        
        # 彎曲的其他手指
        for dx, dy in [(- 25, size//2 + 30), (25, size//2 + 30)]:
            cv2.ellipse(frame, (cx + dx, cy + dy), (12, 20), 0, 0, 360, (180, 200, 255), -1)
            cv2.ellipse(frame, (cx + dx, cy + dy), (12, 20), 0, 0, 360, (100, 120, 200), 2)
        
        # 拇指
        cv2.ellipse(frame, (cx - size + 10, palm_y - 10), (15, 25), -30, 0, 360, (180, 200, 255), -1)
        cv2.ellipse(frame, (cx - size + 10, palm_y - 10), (15, 25), -30, 0, 360, (100, 120, 200), 2)
        
        # 可愛表情（在手掌上）
        cv2.circle(frame, (cx - 12, palm_y - 5), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 12, palm_y - 5), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx - 12, palm_y - 7), 2, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 12, palm_y - 7), 2, (255, 255, 255), -1)
        # 眨眼笑
        cv2.line(frame, (cx - 8, palm_y + 8), (cx + 8, palm_y + 8), (50, 50, 50), 2)

    def draw_gesture(self, frame, gesture, cx, cy, size=60):
        """繪製手勢圖示"""
        if gesture == "rock":
            self.draw_cute_rock(frame, cx, cy, size)
        elif gesture == "paper":
            self.draw_cute_paper(frame, cx, cy, size)
        elif gesture == "scissors":
            self.draw_cute_scissors(frame, cx, cy, size)
        else:
            # 問號
            cv2.putText(frame, "?", (cx - 30, cy + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 150, 150), 5)

    def draw_result_panel(self, frame, x, y, w, h, gesture, label, is_winner=False):
        """繪製結果面板"""
        # 背景
        if is_winner:
            # 金色邊框
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 215, 255), 4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 60), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 120), 2)
        
        # 標籤
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, label, (x + (w - label_size[0]) // 2, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        
        # 手勢圖示（大一點）
        self.draw_gesture(frame, gesture, x + w // 2, y + h // 2 + 20, size=70)
        
        # 手勢名稱
        if gesture:
            name = gesture.upper()
            name_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, name, (x + (w - name_size[0]) // 2, y + h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

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
            
            # 偵測手部（隨時偵測，但只在特定時刻判定）
            frame, current_gesture = self.process_frame(frame)
            
            current_time = time.time()
            
            # ===== 遊戲狀態 =====
            
            if self.game_state == "waiting":
                # 等待畫面
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 250, h//2 - 60),
                             (w//2 + 250, h//2 + 60), (40, 40, 50), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                # 標題
                cv2.putText(frame, "ROCK  PAPER  SCISSORS",
                           (w//2 - 220, h//2 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
                
                cv2.putText(frame, "Press SPACE to Start",
                           (w//2 - 150, h//2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 顯示偵測到的手勢
                if current_gesture:
                    cv2.putText(frame, f"Detected: {current_gesture.upper()}",
                               (w//2 - 80, h//2 + 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            elif self.game_state == "countdown":
                # 倒數計時
                elapsed = current_time - self.countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # 播放倒數音效
                    if remaining != self.last_countdown_num:
                        self.last_countdown_num = remaining
                        play_sound('countdown')
                    
                    # 倒數顯示
                    overlay = frame.copy()
                    cv2.circle(overlay, (w//2, h//2), 100, (40, 40, 50), -1)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                    
                    # 數字
                    cv2.putText(frame, str(remaining),
                               (w//2 - 35, h//2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
                    
                    cv2.putText(frame, "Get Ready!",
                               (w//2 - 70, h//2 + 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                elif remaining == 0 and self.last_countdown_num != 0:
                    # GO! 的瞬間
                    self.last_countdown_num = 0
                    play_sound('go')
                    
                    # 延遲一小段時間後判定（讓玩家反應）
                    self.game_state = "judging"
                    self.judge_start = current_time
            
            elif self.game_state == "judging":
                # 判定階段（短暫延遲後判定）
                elapsed = current_time - self.judge_start
                
                # 顯示 GO!
                overlay = frame.copy()
                cv2.circle(overlay, (w//2, h//2), 100, (40, 40, 50), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame, "GO!",
                           (w//2 - 60, h//2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 100), 5)
                
                # 0.5 秒後判定
                if elapsed > 0.5:
                    self.player_gesture = current_gesture
                    self.ai_gesture = self.get_ai_move(self.player_gesture)
                    self.game_state = "result"
                    self.result_start = current_time
                    
                    if self.player_gesture:
                        self.score["ai"] += 1
                        play_sound('win')
                        print(f"  你: {self.player_gesture} vs AI: {self.ai_gesture} → AI 勝！")
                    else:
                        print("  未偵測到手勢")
            
            elif self.game_state == "result":
                # 結果畫面
                panel_w, panel_h = 200, 220
                gap = 80
                
                # 玩家面板
                player_x = w // 2 - panel_w - gap // 2
                self.draw_result_panel(frame, player_x, h//2 - panel_h//2,
                                      panel_w, panel_h, self.player_gesture, "YOU")
                
                # VS
                cv2.putText(frame, "VS",
                           (w//2 - 25, h//2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # AI 面板（贏家）
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
                
                text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                cv2.putText(frame, result_text,
                           (w//2 - text_size[0]//2, h//2 - panel_h//2 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # 3 秒後回到等待
                if current_time - self.result_start > 3:
                    self.game_state = "waiting"
                    self.last_countdown_num = 0
            
            # ===== 固定 UI =====
            
            # 分數板
            cv2.rectangle(frame, (0, 0), (280, 50), (40, 40, 50), -1)
            cv2.putText(frame, f"AI: {self.score['ai']}  |  You: {self.score['player']}",
                       (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # 底部提示
            cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 50), -1)
            cv2.putText(frame, "SPACE: Start  |  Q: Quit",
                       (w//2 - 130, h - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
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
        if SOUND_ENABLED:
            pygame.mixer.quit()
        
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
        print("  pip install mediapipe opencv-python numpy pygame")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
