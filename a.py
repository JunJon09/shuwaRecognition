import cv2
import mediapipe as mp

# MediaPipeのモジュールを初期化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# PoseとHandsモデルを初期化
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# 動画ファイルを読み込む
cap = cv2.VideoCapture('14.mp4')

# 動画の保存設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output.MP4', fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipeモデルへの入力のために色空間を変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Poseの検出
    pose_results = pose.process(frame_rgb)

    # Handsの検出
    hands_results = hands.process(frame_rgb)

    # 検出結果の描画
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 処理後のフレームを動画ファイルに書き込む
    out.write(frame)

    # フレームを表示
    cv2.imshow('MediaPipe Pose and Hands', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
