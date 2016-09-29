import sys
import numpy as np
import cv2
from math import sqrt
import time
from operator import itemgetter

def image2vector_list(file_name,
                      RESOLUTION=32,
                      AREA_SIZE=10,
                      CHAR_HEIGHT=10,
                      MARGIN=5,
                      X_SCALE=1,
                      Y_SCALE=15,
                      SHOW_LOOP=False,
                      SHOW_RECOGNIZED=False,
                      SHOW_VECTORS=False):

    image = cv2.imread(file_name)

    #### 전처리 시작

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_bw, (5, 5), 0)
    image_threshhold = cv2.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)
    cleaned = image_threshhold.copy()

    #### 전처리 끝

    #### 오브젝트 탐색

    _, contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    countours_filtered = filter(lambda contour:cv2.contourArea(contour) > AREA_SIZE and cv2.boundingRect(contour)[3] > CHAR_HEIGHT, contours)
    countours_sorted = sorted(countours_filtered, key=lambda contour:cv2.boundingRect(contour)[0]*X_SCALE + cv2.boundingRect(contour)[1]*Y_SCALE, reverse=False)

    vectors = np.empty((0, RESOLUTION**2))
    keys = [i for i in range(48,58)]

    ex=[0,0,0,0]
    count = 0
    for index, contour in enumerate(countours_sorted):
        [x, y, w, h] = cv2.boundingRect(contour)  # x,y: top-left point w, h: with, height
        [x_ex, y_ex, w_ex, h_ex] = ex
        nested =  x-x_ex >= 0 and y-y_ex>=0 and (x+w)-(x_ex+w_ex) <= 0 and (y+h)-(y_ex+h_ex) <= 0
        go_back = x-x_ex < 0 and abs(y-y_ex) < h*0.8
        # nested =  False

        if not nested and not go_back:
            cv2.rectangle(image, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), (0, 0, 255), 2)  # 사각형 그리기 params: 배경, 시작점, 반대점, RGB, 선두께
            roi = cleaned[y - MARGIN:y + h + MARGIN, x - MARGIN:x + w + MARGIN]  # cleaned 이미지에서 해당 부분 잘라냄
            roismall = cv2.resize(roi, (RESOLUTION, RESOLUTION))  # 잘라낸 이미지 리사이징

            sample = roismall.reshape((1, RESOLUTION**2)).astype(bool)  # roismall 32*32 => 1*1024
            vectors = np.append(vectors, sample, 0)

            if SHOW_LOOP:
                cv2.imshow('norm',image)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)

            ex = [x, y, w, h]
            count += 1

    # SHOW_RECOGNIZED == True 일때만 오브젝트 탐지한 이미지 보여줌
    if SHOW_RECOGNIZED:
        cv2.imshow('original', image)
        print("print esc to continue")
        if 27 == cv2.waitKey(0):
            pass

    # SHOW_VECTORS == True 일때만 character 매트릭스 출력
    if SHOW_VECTORS == True:
        for sample in vectors:
            print("===================")
            for i in range(RESOLUTION):
                for j in range(RESOLUTION):
                    if sample[i*RESOLUTION+j] == True:
                        print("#", end="")
                    else:
                        print(" ", end="")

                print("")

    return vectors


def make_file_path_list(capitals):
    result = []
    for capital in capitals:
        result.append("train_image/" + capital + ".png")
    return result


def make_train_set(file_path, flag):
    vector_list = image2vector_list(file_path,
                                    Y_SCALE=9,
                                    AREA_SIZE=12,
                                    CHAR_HEIGHT=12,
                                    SHOW_LOOP=False,
                                    SHOW_RECOGNIZED=False,
                                    SHOW_VECTORS=False,
                                    RESOLUTION=32)

    result = []
    vector_list = vector_list.tolist()
    for vector in vector_list:
        result.append([vector, flag])  # 최적화 가능 (to_list for 문 앞으로 빼기)

    return result


def most_common(input_list):
    elems = tuple(input_list)
    biggest_count = 0
    most_common_elem = ""
    for elem in elems:
        count = input_list.count(elem)
        if count > biggest_count:
            biggest_count = count
            most_common_elem = elem
    return most_common_elem


def get_euclid(trained_char_vec, input_char_vector):
    ssq = 0
    for i, px in enumerate(input_char_vector):  # 매 픽셀마다 거리 계산해서 ssq 합산
        ssq += (abs(px-trained_char_vec[i]))  # 최적화 가능. 어차피 1이니까 제곱 불필요
    return sqrt(ssq)


def ocr(file_path, trained_list, k, SHOW_PROGRESS=False):
    input_vectors = image2vector_list(file_path, SHOW_LOOP=False, SHOW_RECOGNIZED=False).tolist()

    result_str = ""

    for input_char_vector in input_vectors:  # 추출된 각 문자마다
        proximate_k = []  # [["A", 10],["A", 20], ...]
        for trained_char in trained_list:  # 모든 학습셋 문자들에 대해서 비교
            euclid = get_euclid(trained_char[0], input_char_vector)

            if len(proximate_k) < k:
                proximate_k.append([trained_char[-1], euclid])
            else:
                proximate_k = sorted(proximate_k, key=lambda x:x[1])
                proximate_k[-1] = [trained_char[-1], euclid]

        votes = [x[0] for x in proximate_k]
        elect = most_common(votes)
        if SHOW_PROGRESS:
            print(elect)
        result_str += elect

    return result_str


if __name__ == '__main__':
    t1 = time.process_time()

    image_file = 'russell.png'

    chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    training_file_path_list = make_file_path_list(chars)

    trained = []
    for index, path in enumerate(training_file_path_list):  # 문자 하나마다 반복
         trained.extend(make_train_set(path, chars[index]))

    result = ocr("russell_short.png", trained, 10, SHOW_PROGRESS=True)
    print(result)
    t2 = time.process_time()
    print("process time: " + str(t2-t1))
