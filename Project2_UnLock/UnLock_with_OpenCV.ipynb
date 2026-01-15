import cv2
import numpy as np
from tensorflow import keras
import time

my_model = keras.models.load_model("my_first_DNN_model.keras")
# my_model.summary()
# predict_image = load_model.predict(rand_test_image[np.newaxis,:,:])
# print(predict_image.argmax())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캡을 열 수 없습니다.")
    exit()


PW = []
Question = []

# 'PW1.txt' 파일열기.
with open("PW1.txt", "r", encoding="utf-8") as f:
    # 1. 첫 번째 줄 읽어오기 (비밀번호)
    first_line = f.readline().strip() # "0,2,6,3"
    if first_line:
        # 콤마(,)를 기준으로 나누고, 숫자로 변환 후 PW 리스트에 담기
        PW = [int(p) for p in first_line.split(',')]

    # 2. 나머지 줄 읽어오기 (문제들)
    for line in f:
        line = line.strip() # 양 끝 공백 및 개행문자(\n) 제거
        if line: # 빈 줄이 아니면
            Question.append(line)

PW_sequence = 0
PW_len = len(PW)

# print(PW)
# print(Question)

# 시작을 구분하는 인자.
Start_f = False
# 시작 시간을 담을 변수 start_time
start_time = time.time()

while True:
    ret, frame = cap.read() # ret(return 값은 True or False)
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    fliped_frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape  # height, width, color channel

    if not Start_f:
        (ST_w, ST_h), _ = cv2.getTextSize("Press the 's' key to start.", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        point_ST = ((width - ST_w) // 2, (height + ST_h) // 2)
        cv2.putText(fliped_frame, "Press the 's' key to start.", point_ST, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam", fliped_frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        Start_f = True
        start_time = time.time()

    if Start_f:
        # 시간 측정 시작.
        elapsed_time = time.time() - start_time

        # Center 좌표 정의.
        center_x, center_y = width // 2, height // 2

        # file로부터 읽어 온 문제 출력.
        QT = f"Q{PW_sequence+1}. {Question[PW_sequence]}"
        (QT_w, QT_h), _ = cv2.getTextSize(QT, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        point_QT = ((width - QT_w) // 2, center_y + 160 + QT_h)
        cv2.putText(fliped_frame, QT, point_QT, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 비밀번호 자릿수 출력.
        DT = f"PW_Digits: {PW_len}"
        (DT_w, DT_h), _ = cv2.getTextSize(DT, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        point_DT = ((width - DT_w - 10), 35)
        cv2.putText(fliped_frame, DT, point_DT, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 현재까지 맞춘 비밀번호 출력.
        cv2.putText(fliped_frame, str(PW[:PW_sequence]), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Timer 출력.
        TT = f"Timer: {elapsed_time:.2f}s"
        cv2.putText(fliped_frame, TT, (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Capture 할 roi 영역 설정 및 표시.
        roi = fliped_frame[center_y-150:center_y + 150, center_x-150:center_x + 150]
        cv2.rectangle(fliped_frame, (center_x-150, center_y-150), (center_x + 150, center_y + 150), (0,0,255), 2)   # 빨간 사각형을 정 가운데 그림

        cv2.imshow("Webcam", fliped_frame)


        # 화면 캡쳐를 위한 키 값 받기.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c' or 'C'):
            # # Color channel 변경 | Flip | Noise 제거
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)      # 이진화를 위한 Color Channel 변경.
            gray_image = np.flip(gray_image, 1)                     # 사용자에게 친숙하게 flip 된 image를 다시 flip
            cv2.imwrite("gray_image.png", gray_image)
            gaussian_blur = cv2.GaussianBlur(gray_image, (5,5), 3)  # Gaussian blur를 이용해 영상 노이즈 제거.

            # 이진화
            _, otsu_thread = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow("otsu_thread", otsu_thread)

            # 인식률 향상을 위한 Morph
            kernel = np.ones((5,5), np.uint8)
            erosion = cv2.erode(otsu_thread, kernel, iterations=5)  # eroding 연산을 5번 반복하여 글자를 두껍게 만듦.
            cv2.imshow("erosion", erosion)
            cv2.imwrite("digit_binary_image.png", erosion)

            # 인식률 향상을 위한 image cropping
            img = cv2.imread("digit_binary_image.png", cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            crop_size = 280
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            x1, x2 = cx-half, cx+half
            y1, y2 = cy-half, cy+half

            # 경계면 설정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_img = img[y1:y2, x1:x2]
            cv2.imshow("cropped_img", cropped_img)

            # image reversing
            reversed_img = cv2.bitwise_not(cropped_img)
            cv2.imshow("reversed_img", reversed_img)
            # cv2.imwrite(f"IMAGE_FOR_TEST({PW_sequence}).png", reversed_img)

            # image resizing
            resized_img = cv2.resize(reversed_img, (28, 28))
            print(resized_img)
            # print(type(resized_img))    # numpy.ndarray
            # print(resized_img.shape)    # (28,28)

            # image Normalizing
            normalized_img = resized_img / 255

            # Predict with my_model
            predict_image = my_model.predict(normalized_img[np.newaxis,:,:])
            # print(predict_image.argmax())

            # 인식된(model에 의해 예측된) 숫자가 현재 자리의 PW 값과 같다면,
            if PW[PW_sequence] == predict_image.argmax():
                # Correct! 문구 출력.
                (CT_w, CT_h), _ = cv2.getTextSize("Correct!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                pointCT = ((width - CT_w) // 2, (height + CT_h) // 2)
                cv2.putText(fliped_frame, "Correct!", pointCT, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 화면 출력 및 출력 화면 저장.
                cv2.imshow("Webcam", fliped_frame)
                cv2.imwrite(f"Question({PW_sequence+1}).png", fliped_frame)
                cv2.waitKey(500)
                PW_sequence += 1    # PW_sequence 변수 변경
                print("Correct.")

                if PW_sequence == PW_len:    # 모든 자리의 PW를 맞췄을 경우
                    # 출력할 검은 화면 설정 (clear 화면)
                    black_screen = np.zeros_like(frame)
                    # PW 자릿수 출력
                    DT = f"PW_Digits: {PW_len}"
                    cv2.putText(black_screen, f"PW_Digits: {PW_len}", point_DT, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Timer 출력
                    TT = f"Timer: {elapsed_time:.2f}s"
                    cv2.putText(black_screen, f"Timer: {elapsed_time:.2f}s", (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # "UnLock!" 문구와 PW 출력
                    (ULT_w, ULT_h), _ = cv2.getTextSize("Unlock!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    pointULT = ((width - ULT_w) // 2, (height + ULT_h) // 2)
                    cv2.putText(black_screen, "UnLock!", pointULT, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(black_screen, str(PW[:PW_sequence]), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Webcam", black_screen)
                    cv2.imwrite(f"UnLock({PW[:]}).png", black_screen)
                    print("Unlock")
                    break
            else:
                # 인식된 숫자와 함게 is not correct 문구 출력.
                NCT = f"'{predict_image.argmax()}' is not Correct."
                (NCT_w, NCT_h), _ = cv2.getTextSize(NCT, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                pointNCT = ((width - NCT_w) // 2, (height + NCT_h) // 2)
                cv2.putText(fliped_frame, NCT, pointNCT, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Webcam", fliped_frame)
                cv2.waitKey(1000)
                print("Wrong.")
                continue

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()

while True:
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
