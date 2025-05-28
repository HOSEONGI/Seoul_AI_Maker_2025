import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 💾 모델 및 클래스 로드
model = load_model("quickdraw_5class_model.keras")
with open("categories.txt", "r") as f:
    class_names = f.read().splitlines()

# 🎥 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 설정 (화면 중앙 기준, 사이즈 확대: 300x300)
    h, w, _ = frame.shape
    # 가로가 긴 직사각형 ROI (화면 중앙 기준)
    roi_width = int(w * 0.5)
    roi_height = int(h * 0.6)
    x1 = w // 2 - roi_width // 2
    y1 = h // 2 - roi_height // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    # ROI 추출
    roi = frame[y1:y2, x1:x2]

    # 전처리: 흑백 + 반전 + 리사이즈 + 정규화
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (28, 28))
    normalized = resized / 255.0
    input_img = normalized.reshape(1, 28, 28, 1)

    # 예측 전: 빈 화면 판별
    if np.sum(input_img) < 10:
        predicted_label = "No Drawing"
    else:
        prediction = model.predict(input_img, verbose=0)
        confidence = np.max(prediction)

        if confidence < 0.7:
            predicted_label = "I don't know this"
        else:
            predicted_label = class_names[np.argmax(prediction)]
            predicted_label += f" ({confidence*100:.1f}%)"

        # 화면 출력
        # 화면 출력
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 텍스트 설정
    text = f"Answer: {predicted_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 텍스트 위치: ROI 상단 중앙
    text_x = x1 + (roi_width - text_size[0]) // 2
    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    cv2.imshow("AI Drawing Inference", frame)

    # 종료 키: 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()