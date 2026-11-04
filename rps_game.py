#!/usr/bin/env python3
"""
剪刀石頭布遊戲 - AI 永遠贏版本
左右分割畫面，使用自訂圖片

安裝：
    pip install mediapipe opencv-python numpy

執行：
    python rps_game.py
    
注意：assets 資料夾需要有 rock.png, paper.png, scissors.png
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import subprocess
import threading
import argparse
import random

# 全域變數
mp = None
landmarker = None


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='剪刀石頭布遊戲 - AI 永遠贏版本')
    parser.add_argument('--win', type=float, default=0.0,
                       help='使用者贏的機率 (0.0-1.0)，預設 0.0 表示 AI 永遠贏')
    args = parser.parse_args()
    
    # 確保機率在合理範圍
    args.win = max(0.0, min(1.0, args.win))
    return args


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
    def __init__(self, win_rate=0.0):
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
        
        # 使用者贏的機率
        self.win_rate = win_rate
        
        # 輪播動畫
        self.slot_index = 0
        self.slot_gestures = ["rock", "paper", "scissors"]
        self.last_slot_time = 0
        
        # 手勢偵測緩衝
        self.gesture_samples = []
        self.detection_duration = 0.3  # 偵測時間
        
        # 贏法對照表（AI 要出什麼才能贏）
        self.winning_move = {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }
        
        # 輸法對照表（AI 要出什麼才會輸）
        self.losing_move = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper"
        }
        
        # 顏色
        self.ai_bg_color = (60, 60, 220)  # 紅色 (BGR)
        
        # 上一局結果
        self.last_result = None
        
        # 載入圖片
        self.load_images()

    def load_images(self):
        """載入手勢圖片並處理透明背景"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir, "assets")
        
        self.images = {}
        for gesture in ["rock", "paper", "scissors"]:
            img_path = os.path.join(assets_dir, f"{gesture}.png")
            if os.path.exists(img_path):
                # 讀取圖片（含透明通道）
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # 處理透明背景 - 加上白色背景
                    if img.shape[2] == 4:
                        # 分離 BGR 和 Alpha
                        bgr = img[:, :, :3]
                        alpha = img[:, :, 3] / 255.0
                        
                        # 建立白色背景
                        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                        
                        # 混合圖片和白色背景
                        for c in range(3):
                            white_bg[:, :, c] = (
                                alpha * bgr[:, :, c] + 
                                (1 - alpha) * white_bg[:, :, c]
                            ).astype(np.uint8)
                        
                        self.images[gesture] = white_bg
                    else:
                        self.images[gesture] = img
                    print(f"  ✓ 載入 {gesture}.png")
                else:
                    print(f"  ✗ 無法讀取 {gesture}.png")
            else:
                print(f"  ✗ 找不到 {img_path}")
        
        if len(self.images) < 3:
            print("\n⚠️  部分圖片載入失敗，將使用內建圖形")

    def overlay_image(self, background, img, x, y, size):
        """將圖片疊加到背景上"""
        if img is None:
            return
        
        # 調整圖片大小
        img_resized = cv2.resize(img, (size, size))
        
        # 計算放置位置（置中）
        x1 = x - size // 2
        y1 = y - size // 2
        x2 = x1 + size
        y2 = y1 + size
        
        # 邊界檢查
        bg_h, bg_w = background.shape[:2]
        
        # 計算實際可用範圍
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = size - max(0, x2 - bg_w)
        src_y2 = size - max(0, y2 - bg_h)
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(bg_w, x2)
        dst_y2 = min(bg_h, y2)
        
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return
        
        # 直接覆蓋
        background[dst_y1:dst_y2, dst_x1:dst_x2] = img_resized[src_y1:src_y2, src_x1:src_x2]

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
        """AI 出招 - 根據機率決定輸贏"""
        if player_gesture is None:
            return random.choice(self.slot_gestures)
        
        # 根據機率決定 AI 是要贏還是輸
        if random.random() < self.win_rate:
            # 使用者贏 - AI 出會輸的
            return self.losing_move[player_gesture]
        else:
            # AI 贏 - AI 出會贏的
            return self.winning_move[player_gesture]

    def get_most_common_gesture(self):
        """從收集的樣本中取最常出現的手勢"""
        if not self.gesture_samples:
            return None
        
        valid_samples = [g for g in self.gesture_samples if g is not None]
        if not valid_samples:
            return None
        
        counts = {"rock": 0, "paper": 0, "scissors": 0}
        for g in valid_samples:
            counts[g] += 1
        
        return max(counts, key=counts.get)

    def draw_gesture(self, frame, gesture, cx, cy, size=150):
        """繪製手勢圖示"""
        if gesture in self.images:
            self.overlay_image(frame, self.images[gesture], cx, cy, size)
        else:
            # 備用：畫問號
            cv2.putText(frame, "?", (cx - 30, cy + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)

    def run(self):
        """主遊戲循環"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("\n❌ 無法開啟鏡頭！")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 50)
        print("🎮 剪刀石頭布遊戲")
        if self.win_rate > 0:
            print(f"   使用者勝率: {self.win_rate * 100:.0f}%")
        else:
            print("   模式: AI 永遠贏")
        print("=" * 50)
        print("  SPACE/ENTER = 開始")
        print("  Q/ESC = 離開")
        print("  ⚠️  請先點擊遊戲視窗才能使用鍵盤！")
        print("=" * 50)
        
        # 視窗標題
        if self.win_rate > 0:
            self.window_title = f"Rock Paper Scissors - You Win {self.win_rate*100:.0f}%"
        else:
            self.window_title = "Rock Paper Scissors - AI Always Wins"
        
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
            
            # ===== 左半邊：玩家（全螢幕鏡頭）=====
            cam_h, cam_w = camera_frame.shape[:2]
            # 裁切並縮放以填滿左半邊
            cam_aspect = cam_w / cam_h
            target_aspect = half_width / window_height
            
            if cam_aspect > target_aspect:
                # 鏡頭太寬，裁切左右
                new_w = int(cam_h * target_aspect)
                start_x = (cam_w - new_w) // 2
                cropped = camera_frame[:, start_x:start_x + new_w]
            else:
                # 鏡頭太高，裁切上下
                new_h = int(cam_w / target_aspect)
                start_y = (cam_h - new_h) // 2
                cropped = camera_frame[start_y:start_y + new_h, :]
            
            resized_cam = cv2.resize(cropped, (half_width, window_height))
            frame[0:window_height, 0:half_width] = resized_cam
            
            # 玩家分數（半透明背景）
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (180, 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"YOU: {self.score['player']}", (15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # ===== 右半邊：AI =====
            cv2.rectangle(frame, (half_width, 0), (window_width, window_height), 
                         self.ai_bg_color, -1)
            
            # AI 分數
            cv2.putText(frame, f"COMPUTER: {self.score['ai']}", (half_width + 15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            current_time = time.time()
            
            # AI 圖示位置
            ai_cx = half_width + half_width // 2
            ai_cy = window_height // 2
            
            # ===== 遊戲狀態 =====
            
            if self.game_state == "waiting":
                # 等待畫面
                # 玩家側：顯示偵測到的手勢（放大）
                if current_gesture:
                    self.draw_gesture(frame, current_gesture, half_width // 2, window_height - 120, 140)
                
                # AI 側：問號
                cv2.putText(frame, "?", (ai_cx - 50, ai_cy + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 10)
                
                # 中間 VS
                cv2.putText(frame, "VS", (half_width - 35, window_height // 2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 提示
                overlay = frame.copy()
                cv2.rectangle(overlay, (window_width//2 - 180, window_height - 70),
                             (window_width//2 + 180, window_height - 20), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "Press SPACE to Start!", 
                           (window_width // 2 - 150, window_height - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            elif self.game_state == "countdown":
                # 倒數計時
                elapsed = current_time - self.countdown_start
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    if remaining != self.last_countdown_num:
                        self.last_countdown_num = remaining
                        play_countdown_sound()
                    
                    # 玩家側顯示手勢（放大）
                    if current_gesture:
                        self.draw_gesture(frame, current_gesture, half_width // 2, window_height - 120, 140)
                    
                    # AI 輪播（放大）
                    if current_time - self.last_slot_time > 0.1:
                        self.slot_index = (self.slot_index + 1) % 3
                        self.last_slot_time = current_time
                    
                    slot_gesture = self.slot_gestures[self.slot_index]
                    self.draw_gesture(frame, slot_gesture, ai_cx, ai_cy, 220)
                    
                    # 倒數數字
                    cv2.putText(frame, str(remaining), 
                               (half_width - 45, window_height // 2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                
                elif remaining == 0 and self.last_countdown_num != 0:
                    self.last_countdown_num = 0
                    play_go_sound()
                    self.game_state = "judging"
                    self.judge_start = current_time
                    self.gesture_samples = []
            
            elif self.game_state == "judging":
                # 判定階段
                elapsed = current_time - self.judge_start
                
                # 收集手勢樣本
                if current_gesture:
                    self.gesture_samples.append(current_gesture)
                
                # AI 輪播變慢（放大）
                slot_speed = 0.08 + (elapsed / self.detection_duration) * 0.1
                if current_time - self.last_slot_time > slot_speed:
                    self.slot_index = (self.slot_index + 1) % 3
                    self.last_slot_time = current_time
                
                slot_gesture = self.slot_gestures[self.slot_index]
                self.draw_gesture(frame, slot_gesture, ai_cx, ai_cy, 220)
                
                # GO!
                cv2.putText(frame, "GO!", (half_width - 60, window_height // 2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 100), 6)
                
                # 玩家手勢（放大）
                if current_gesture:
                    self.draw_gesture(frame, current_gesture, half_width // 2, window_height - 120, 140)
                
                # 時間到
                if elapsed >= self.detection_duration:
                    self.player_gesture = self.get_most_common_gesture()
                    self.ai_gesture = self.get_ai_move(self.player_gesture)
                    self.game_state = "result"
                    self.result_start = current_time
                    
                    if self.player_gesture:
                        # 判斷誰贏
                        if self.winning_move[self.player_gesture] == self.ai_gesture:
                            # AI 贏
                            self.score["ai"] += 1
                            self.last_result = "ai"
                            print(f"  你: {self.player_gesture} vs AI: {self.ai_gesture} → AI 勝！")
                        else:
                            # 玩家贏
                            self.score["player"] += 1
                            self.last_result = "player"
                            print(f"  你: {self.player_gesture} vs AI: {self.ai_gesture} → 你贏了！")
                        play_win_sound()
                    else:
                        self.last_result = "none"
                        print("  未偵測到手勢")
            
            elif self.game_state == "result":
                # 結果畫面
                # 玩家手勢（放大）
                if self.player_gesture:
                    self.draw_gesture(frame, self.player_gesture, half_width // 2, window_height // 2 + 30, 200)
                else:
                    cv2.putText(frame, "?", (half_width // 2 - 40, window_height // 2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (150, 150, 150), 8)
                
                # AI 手勢（放大）
                self.draw_gesture(frame, self.ai_gesture, ai_cx, ai_cy, 250)
                
                # VS
                cv2.putText(frame, "VS", (half_width - 35, window_height // 2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 結果
                if self.player_gesture:
                    if self.last_result == "player":
                        result_text = "YOU WIN!"
                        result_color = (0, 255, 0)  # 綠色
                    else:
                        result_text = "AI WINS!"
                        result_color = (0, 255, 255)  # 黃色
                    
                    # 閃爍
                    if int(current_time * 4) % 2 == 0:
                        cv2.putText(frame, result_text,
                                   (window_width // 2 - 120, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, result_color, 4)
                else:
                    cv2.putText(frame, "No Hand Detected",
                               (window_width // 2 - 150, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                
                # 3 秒後回到等待
                if current_time - self.result_start > 3:
                    self.game_state = "waiting"
                    self.last_countdown_num = 0
            
            # ===== 分隔線 =====
            cv2.line(frame, (half_width, 0), (half_width, window_height), (255, 255, 255), 3)
            
            # 顯示
            cv2.imshow(self.window_title, frame)
            
            # 鍵盤
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            
            try:
                if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
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
    args = parse_args()
    
    print("🎮 剪刀石頭布遊戲")
    if args.win > 0:
        print(f"   使用者勝率: {args.win * 100:.0f}%")
    else:
        print("   模式: AI 永遠贏")
    print("正在初始化...\n")
    
    try:
        setup_mediapipe()
        game = RockPaperScissorsGame(win_rate=args.win)
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
