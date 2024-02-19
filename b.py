import mediapipe as mp
import cv2

# MediaPipe Face Meshモデルの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# 画像の読み込み
path = "a.png"
image = cv2.imread(path)

# 画像が読み込めたか確認
if image is None:
    print(f"Unable to load image '{path}'")
else:
    # BGR画像をRGBに変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 画像をMediaPipeに渡す
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                print(f'Landmark {idx}: ({x}, {y})')
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # 画像の表示
    cv2.imshow('MediaPipe Face Mesh', image)
    cv2.waitKey(0)

# 画像ウィンドウを閉じる
cv2.destroyAllWindows()
