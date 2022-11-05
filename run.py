from Realtime import Realime_graph
from PyQt5.QtCore import QTimer
import sys
import pymysql
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QColor
from subprocess import Popen, PIPE
from PIL import Image
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
import os
import itertools
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

sys.path.insert(0, './yolov5')

# PYQT 카메라 ON,OFF 버튼 선택
running = True

# PYQT 직사각형, 폴리곤 ROI 라디오 버튼 선택
Choose_pyqt_Rect = False
Choose_pyqt_Polygon = False

# 폴리곤, 직사각형, ROI 설정 초기화
redrectangle_roi_pyqt = False
redpolygon_roi_pyqt = False

# ROI Settings 윈도우창 ESC버튼 누른 순간 ROI 모드 활성화
roi_mode_on = False

# 마우스 상태 및 직사각형 ROI 좌표, 폴리곤 좌표 리스트 초기화
mouse_is_pressing, step = False, 0
start_x, start_y, end_x, end_y = 0,0,0,0
distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y = 0, 0 ,0 ,0
polygon_xy_list = []

# 행동 인식 딥러닝 모델 선택 초기화
action_mode = "disable"





# 제한구역 직사각형 ROI 버튼 설정 및 마우스 상태 초기화
restricted_area_redrectangle_roi_pyqt = False
distance_roi_step = 0

# 충돌사고, 제한구역 거리 모드 초기화
distance_mode = False
roi_distance_mode = False

# PYQT 안전보호구 모드 초기화
safety_mode_on = False

# PYQT 정보 출력 장치 라디오 버튼 선택
Choose_pyqt_pc = False
Choose_pyqt_smartphone = False
Choose_pyqt_disable = True

# 실시간 그래프 - deepsort 코드에서 인원수 가져오기 위한 리스트
count_graph = []

# 실시간 그래프 -  거리에 따른 "low risk, high risk" 리스트
distance_graph = []
restricted_distance_graph = []
safety_distance_graph = []

# Warning Action -> Dangerous Action 라벨 변환 큐 리스트
fw_queue = []

# 파이어베이스 정보 및 이미지 전송 제약 시간 변수
tm = 0
tm2 = 0

# 파이어베이스 이미지 전송 파일 경로 및 이름 초기화
cred = credentials.Certificate('saem-29570-firebase-adminsdk-a1n4v-74ab4b490b.json')
firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://saem-29570-default-rtdb.firebaseio.com'
})
fire_filename = "capture.png"
bucket = storage.bucket('saem-29570.appspot.com')


# 제한구역와 객체 간의 거리측정
def restricted_area_distancing(people_coords, img, distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y, dist_thres_lim=(250, 10000)):
    global restricted_distance_graph

    distance_high_count = 0
    distance_low_count = 0

    # restricted_area roi 영역 그리기
    ROI_box = img[distance_roi_st_y: distance_roi_end_y, distance_roi_st_x: distance_roi_end_x]
    ROI_box = cv2.add(ROI_box, (200, 200, 200, 0))
    img[distance_roi_st_y: distance_roi_end_y, distance_roi_st_x: distance_roi_end_x] = ROI_box
    cv2.rectangle(img, (distance_roi_st_x, distance_roi_st_y), (distance_roi_end_x, distance_roi_end_y), (255, 255, 255), 3)

    a = torch.tensor(distance_roi_st_x, device="cuda:0")
    b = torch.tensor(distance_roi_st_y, device="cuda:0")
    c = torch.tensor(distance_roi_end_x, device="cuda:0")
    d = torch.tensor(distance_roi_end_y, device="cuda:0")

    roi_xyxy = list()
    roi_xyxy.append(a)
    roi_xyxy.append(b)
    roi_xyxy.append(c)
    roi_xyxy.append(d)

    # Plot lines connecting people
    already_red = dict()  # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []

    for i in people_coords:
        centers.append(((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2))
    roi_centers = ((int(a) + int(c)) // 2, (int(b) + int(d)) // 2)
    centers.append(roi_centers)

    for j in centers:
        already_red[j] = 0
    x_combs = list()

    for i in range(len(people_coords)):
        x_combs.append([people_coords[i], roi_xyxy])
    radius = 10
    thickness = 5

    for x in x_combs:
        xyxy1, xyxy2 = x[0], x[1]
        cntr1 = ((int(xyxy1[2]) + int(xyxy1[0])) // 2, (int(xyxy1[3]) + int(xyxy1[1])) // 2)
        cntr2 = ((int(xyxy2[2]) + int(xyxy2[0])) // 2, (int(xyxy2[3]) + int(xyxy2[1])) // 2)
        dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5

        if dist > dist_thres_lim[0] and dist < dist_thres_lim[1]:
            color = (0, 255, 255)
            label = "Low Risk "
            cv2.rectangle(img, (int(xyxy1[0]), int(xyxy1[1])), (int(xyxy1[2]), int(xyxy1[3])), color, 3)
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                cntr = ((int(xy[2]) + int(xy[0])) // 2, (int(xy[3]) + int(xy[1])) // 2)
                if already_red[cntr] == 0:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            distance_low_count += 1
        elif dist < dist_thres_lim[0]:
            color = (0, 0, 255)
            label = "High Risk"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.rectangle(img, (int(xyxy1[0]), int(xyxy1[1])), (int(xyxy1[2]), int(xyxy1[3])), color, 3)

            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            distance_high_count += 1

    restricted_distance_graph.append(distance_low_count)
    restricted_distance_graph.append(distance_high_count)

    return [distance_low_count, distance_high_count]


# 객체별 거리측정
def distancing(people_coords, img, dist_thres_lim=(250, 550)):
    global distance_graph

    distance_high_count = 0
    distance_low_count = 0

    # Plot lines connecting people
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []

    for i in people_coords:
        centers.append(((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2))

    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.combinations(people_coords, 2))
    radius = 10
    thickness = 5

    for x in x_combs:
        xyxy1, xyxy2 = x[0], x[1]
        cntr1 = ((int(xyxy1[2]) + int(xyxy1[0])) // 2, (int(xyxy1[3]) + int(xyxy1[1])) // 2)
        cntr2 = ((int(xyxy2[2]) + int(xyxy2[0])) // 2, (int(xyxy2[3]) + int(xyxy2[1])) // 2)
        dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5

        if dist > dist_thres_lim[0] and dist < dist_thres_lim[1]:
            color = (0, 255, 255)
            label = "Low Risk "
            cv2.line(img, cntr1, cntr2, color, thickness)
            if already_red[cntr1] == 0:
                cv2.circle(img, cntr1, radius, color, -1)
            if already_red[cntr2] == 0:
                cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                cntr = ((int(xy[2]) + int(xy[0])) // 2, (int(xy[3]) + int(xy[1])) // 2)
                if already_red[cntr] == 0:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            distance_low_count += 1
        elif dist < dist_thres_lim[0]:
            color = (0, 0, 255)
            label = "High Risk"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            distance_high_count += 1

    # PYQT 그래프에 나타낼 low risk 갯수, high risk 갯수 리스트에 저장
    distance_graph.append(distance_low_count)
    distance_graph.append(distance_high_count)

    return [distance_low_count, distance_high_count]


# 직사각형 ROI 마우스 이벤트 핸들러 함수, 좌푯값 저장
def Mouse_Callback_Rect_One(event, x, y, flags, params):
    global step , start_x, end_x, start_y, end_y, mouse_is_pressing

    # Press The Left Button
    if event == cv2.EVENT_LBUTTONDOWN :
        step = 1
        mouse_is_pressing = True
        start_x = x
        start_y = y
    # Moving The Mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        # If Pressing The Mouse
        if mouse_is_pressing:
            step = 2
            end_x = x
            end_y = y
    # Release The Left Button
    elif event == cv2.EVENT_LBUTTONUP:
        step = 3
        mouse_is_pressing = False
        end_x = x
        end_y = y
    else:
        print("Error : Mouse_Callback_Rect_One 함수 예외처리")


# 제한구역 직사각형 ROI 마우스 이벤트 핸들러 함수, 좌푯값 저장
def Mouse_Callback_Rect_Two(event, x, y, flags, params):
    global distance_roi_step , distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y, mouse_is_pressing

    # Press The Left Button
    if event == cv2.EVENT_LBUTTONDOWN :
        distance_roi_step = 1
        mouse_is_pressing = True
        distance_roi_st_x = x
        distance_roi_st_y = y
    # Moving The Mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        # If Pressing The Mouse
        if mouse_is_pressing:
            distance_roi_step = 2
            distance_roi_end_x = x
            distance_roi_end_y = y
    # Release The Left Button
    elif event == cv2.EVENT_LBUTTONUP:
        distance_roi_step = 3
        mouse_is_pressing = False
        distance_roi_end_x = x
        distance_roi_end_y = y
    else:
        print("Error : Mouse_Callback_Rect_Two 함수 예외처리")


# 폴리곤 ROI 마우스 이벤트 핸들러 함수, 좌푯값 저장
def Mouse_Callback_Polygon(event, x, y, flags, params):
    # Press The Left Button
    global polygon_xy_list, step

    if event == cv2.EVENT_LBUTTONDOWN:
        step = 100
        xy_list = [x, y]
        polygon_xy_list.append(xy_list)
    elif event == cv2.EVENT_RBUTTONDOWN:
        step = 200
    elif event == cv2.EVENT_MBUTTONDOWN:
        step = 0
        polygon_xy_list.clear()
    else:
        print("Error : Mouse_Callback_Proygon 함수 예외처리")


# 직사각형 ROI 그리기 및 좌표값 변환 함수, 만약 Roi Mode 활성화 시 직사각형이 사라짐
def draw_roi_rectangle(img, step, start_x, end_x, start_y, end_y):
    # Click The Mouse Button
    if step == 1:
        cv2.circle(img, (start_x, start_y), 10, (0, 255, 0), -1)
    # Moving The Mouse
    elif step == 2:
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    # Release Of The Mouse
    elif step == 3:
        # If Start X Position Is Bigger Than End X
        if start_x > end_x and start_y < end_y:  # 오른쪽 위에서 왼쪽 아래로 드래그 할 시
            if end_x < 0:
                end_x = 0
            start_x, end_x = end_x, start_x
        elif start_x > end_x and start_y > end_y:  # 오른쪽 아래서 왼쪽 위로 드래그 할 시
            if end_y < 0:
                end_y = 0
            start_y, end_y = end_y, start_y
            start_x, end_x = end_x, start_x
        elif start_x < end_x and start_y > end_y:  # 왼쪽 아래서 오른쪽 위로 드래그 할 시
            if end_y < 0:
                end_y = 0
            start_y, end_y = end_y, start_y

    return img, start_x, end_x, start_y, end_y


# 폴리곤 ROI 그리기 및 좌표값 변환 함수, 만약 Roi Mode 활성화 시 폴리곤이 사라짐
def draw_roi_polygon(img, step, polygon_xy_list):
    # Click The Mouse Button
    if step == 100:
        np_xy = np.array(polygon_xy_list)
        for i in range(len(np_xy)):
            cv2.circle(img, (np_xy[i][0], np_xy[i][1]), 10, (0, 255, 0), -1)
            cv2.polylines(img, [np_xy], False, (0, 255, 0), 3)
    elif step == 200:
        np_xy = np.array(polygon_xy_list)
        for i in range(len(np_xy)):
            cv2.polylines(img, [np_xy], True, (0, 255, 255), 3)

    return img


# 박스 크기 조절 해주는 함수
def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    global start_x, start_y

    if roi_mode_on == True:
        bbox_left = min([xyxy[0].item(), xyxy[2].item()]) + (start_x)
        bbox_top = min([xyxy[1].item(), xyxy[3].item()]) + (start_y)
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
    elif roi_mode_on == False:
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h

    return x_c, y_c, w, h


# 메인 실행함수
def detect(opt, save_img=False):
    global running, tm2
    global redrectangle_roi_pyqt, redpolygon_roi_pyqt, roi_mode_on, safety_mode_on
    global start_x, start_y, end_x, end_y, polygon_xy_list
    global distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y
    global step, mouse_is_pressing, distance_roi_step
    global Choose_pyqt_Rect, Choose_pyqt_Polygon
    global action_mode, distance_mode, roi_distance_mode, restricted_area_redrectangle_roi_pyqt
    global count_graph, fw_queue, safety_distance_graph
    global  Choose_pyqt_pc, Choose_pyqt_smartphone, Choose_pyqt_disable

    # pyqt start 버튼 누르면 다시 실행 될 수 있도록 True 설정
    running = True

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # 배회 침입 데이터 딕셔너리
    wander = {}

    # Fighting  & Falling_down 유지시간 저장 리스트
    fight_time = [False,0]
    falling_down_time = [False, 0]
    smoking_time = {}

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        save_img = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = view_img
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    vid = cv2.VideoCapture(source)

    # 실시간 영상저장
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 6.0, (1280, 720)) #해상도에 따라 1280 720 맞춰줘야함
    filename = os.path.basename(source).split('.')[0]
    save_path = f"inference/output/{filename}_action.mp4"
    fps = vid.get(cv2.CAP_PROP_FPS)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 영상 저장 처리
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # ffmpeg setup
    pipe = Popen([
        'ffmpeg', '-loglevel', 'quiet', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', f'{fps}',
        '-i', '-', '-vcodec', 'libx264', '-crf', '28', '-preset', 'veryslow', '-framerate', f'{fps}', f'{save_path}'
    ], stdin=PIPE)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length, position=0, leave=True)

    # 여기부터 데이터 읽어와서 반복문 무한루프 진행
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        # pyqt 종료 버튼 누르면 카메라가 작동하지 않도록 프로그램 종료
        if running == False :
            print(running)
            return 0
        start = time.time()

        # ROI Settings 윈도우창을 이용해 마우스 좌푯값을 수정했을때만 실행
        if not (start_x == 0 or end_x == 0 or start_y == 0 or end_y == 0):

            # 직사각형, 폴리곤 메시지박스 선택 했을때
            if Choose_pyqt_Rect == True or Choose_pyqt_Polygon == True:
                # ROI Mode 활성화 및 마우스 좌표 설정이 끝났을 때
                if (roi_mode_on == True) and (mouse_is_pressing==False) :
                    end_y = end_y - ((end_y - start_y) % 32)
                    end_x = end_x - ((end_x - start_x) % 32)
                    if Choose_pyqt_Rect == True:
                        # 실제, 직사각형 ROI 영역 지정
                        img = img[:, :, start_y: end_y, start_x: end_x]
                        print(f"직사각형 ROI 영역 좌푯값 == start_x : {start_x}, start_y : {start_y}, end_x : {end_x}, end_y : {end_y}")
                    else:
                        # 실제, 폴리곤 ROI 영역 지정
                        img = img[:, :, start_y: end_y, start_x: end_x]
                        print(
                            f"폴리곤 ROI 영역 좌푯값 == start_x : {start_x}, start_y : {start_y}, end_x : {end_x}, end_y : {end_y}")
            else:
                print("PYQT 메시지 박스 '직사각형' 과 '폴리곤' 중 선택 하세요")

        print(f'0번째 -->  {img.shape}')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 객체 간 거리측정, 제한구역 함수 인자 및 안전보후구에 사용할 리스트, 감지된 박스 좌표 저장
        people_coords = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, im0 = path, im0s
                # 영일 영상
                # ui = 144
                # io = 1174
                # op = 66
                # pp = 400
                # cv2.rectangle(im0, (ui, op), (io, pp), (0, 0, 255), 3)

            # s += '%gx%g ' % img.shape[2:]  # print string
            # save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # ROI영역 안의 박스크기 조절
                if roi_mode_on == True:
                    im02 = im0[start_y: end_y, start_x: end_x]
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im02.shape).round()
                elif roi_mode_on == False:
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # person class 좌표 리스트
                bbox_xywh = []
                confs = []

                # vest, helmet class 좌표 리스트
                vest_bbox_xywh = []
                helmet_bbox_xywh = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                  # person class만 deepsort 및 행동인식 수행
                  # safety_mode 활성화 시
                  if safety_mode_on:
                      if names[int(cls)] == 'person' :
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        people_coords.append(xyxy)
                      elif names[int(cls)] == 'vest' :
                          v_x_c, v_y_c, v_bbox_w, v_bbox_h = bbox_rel(*xyxy)
                          v_c_obj = [v_x_c, v_y_c]
                          vest_bbox_xywh.append(v_c_obj)
                      elif names[int(cls)][-6:] == 'helmet' :
                          h_x_c, h_y_c, h_bbox_w, h_bbox_h = bbox_rel(*xyxy)
                          h_c_obj = [h_x_c, h_y_c]
                          helmet_bbox_xywh.append(h_c_obj)
                  else :
                      if names[int(cls)] == 'person':
                          x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                          obj = [x_c, y_c, bbox_w, bbox_h]
                          bbox_xywh.append(obj)
                          confs.append([conf.item()])
                          people_coords.append(xyxy)

                # safety_mode 활성화 시
                if safety_mode_on:
                    # vest, helmet 좌표가 person 좌표안에 없으면 세모 또는 X 출력
                    safety_distance_high_count = 0
                    safety_distance_low_count = 0
                    for p_c_xy in bbox_xywh:
                    #for person_num, p_c_xy in enumerate(bbox_xywh):
                        check_vest_or_helmet = 0
                        p_x_c, p_y_c, p_bbox_w, p_bbox_h = p_c_xy
                        p_xx1 = (int(p_x_c) - int((p_bbox_w / 2)), int(p_y_c) - int((p_bbox_h / 2)))
                        p_yy1 = (int(p_x_c) + int((p_bbox_w / 2)), int(p_y_c) + int((p_bbox_h / 2)))
                        p_xx2 = (int(p_x_c) + int((p_bbox_w / 2)), int(p_y_c) - int((p_bbox_h / 2)))
                        p_yy2 = (int(p_x_c) - int((p_bbox_w / 2)), int(p_y_c) + int((p_bbox_h / 2)))

                        if len(vest_bbox_xywh) == 0 and len(helmet_bbox_xywh) == 0 :
                            color = (0, 0, 255)
                            safety_distance_high_count += 1
                            cv2.line(im0, p_xx1, p_yy1, color, thickness=5)
                            cv2.line(im0, p_xx2, p_yy2, color, thickness=5)
                        else :
                            if len(vest_bbox_xywh) != 0:
                                for v_c_xy in vest_bbox_xywh:
                                        if (p_xx1[0] <= int(v_c_xy[0]) <= p_yy1[0]) and (p_xx1[1] <= int(v_c_xy[1]) <= p_yy1[1]):
                                            check_vest_or_helmet += 1
                                            break
                            if len(helmet_bbox_xywh) != 0:
                                for h_c_xy in helmet_bbox_xywh:
                                        if (p_xx1[0] <= int(h_c_xy[0]) <= p_yy1[0]) and (p_xx1[1] <= int(h_c_xy[1]) <= p_yy1[1]):
                                            check_vest_or_helmet += 1
                                            break
                            if check_vest_or_helmet == 2 :
                                # 헬멧+조끼 둘다 올바르게 착용함 : 정상(초록색)
                                pass
                            elif check_vest_or_helmet == 1 :
                                draw_triangle = np.array([[p_xx1[0], p_yy2[1]], [p_yy1[0], p_yy1[1]], [(p_xx1[0] + p_xx2[0]) / 2, p_xx1[1]]], dtype=np.int32)
                                # 헬멧+조끼 둘중 하나만 착용함 : 경고(노란색)
                                safety_distance_low_count += 1
                                cv2.polylines(im0, [draw_triangle], True, (0, 255, 255), thickness=5)
                            elif check_vest_or_helmet == 0:
                                color = (0, 0, 255)
                                # 헬멧+조끼 둘다 착용 안함 : 위험(빨간색)
                                safety_distance_high_count += 1
                                cv2.line(im0, p_xx1, p_yy1, color, thickness=5)
                                cv2.line(im0, p_xx2, p_yy2, color, thickness=5)

                    safety_distance_graph.append(safety_distance_low_count)
                    safety_distance_graph.append(safety_distance_high_count)
                    vest_bbox_xywh.clear()
                    helmet_bbox_xywh.clear()
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # 객체별 거리 측정 후 선을 이어줌
                if distance_mode == True:
                    lh_list = distancing(people_coords, im0)
                    if lh_list is not None:
                        low_counting_text = "Low Risk : {}".format(lh_list[0])
                        cv2.putText(im0, low_counting_text, (300, im0.shape[0] - 25), cv2.LINE_AA, 0.85, (0, 255, 255), 2)
                        high_counting_text = "High Risk : {}".format(lh_list[1])
                        cv2.putText(im0, high_counting_text, (500, im0.shape[0] - 25), cv2.LINE_AA, 0.85, (0, 0, 255), 2)

                # 제한구역에 접근하는 객체 거리 측정 후 경고 박스 생성
                if restricted_area_redrectangle_roi_pyqt == True:
                    cv2.namedWindow("Restricted Area ROI Settings")
                    cv2.setMouseCallback("Restricted Area ROI Settings", Mouse_Callback_Rect_Two)
                    im0, distance_roi_st_x, distance_roi_end_x, distance_roi_st_y, distance_roi_end_y = draw_roi_rectangle(im0, distance_roi_step, distance_roi_st_x, distance_roi_end_x, distance_roi_st_y, distance_roi_end_y)
                    # 마우스 눌렀다 때면 PYQT 창에도 휘발성으로 노란색 직사각형을 그림
                    if distance_roi_step == 3:
                        # restricted_area_redrectangle_roi_pyqt = False # 만약 눌렀다 떈 동시에 윈도우창을 끄고 싶으면
                        cv2.rectangle(im0, (distance_roi_st_x, distance_roi_st_y), (distance_roi_end_x, distance_roi_end_y), (0, 255, 255), 3)
                    cv2.imshow("Restricted Area ROI Settings", im0)
                    key = cv2.waitKey(1)
                    # esc 누를경우, ROI 직사각형 좌표 설정 종료 및 RoI Mode 활성화
                    if key == 27:
                        restricted_area_redrectangle_roi_pyqt = False
                        cv2.destroyWindow("Restricted Area ROI Settings")
                        roi_distance_mode = True
                        mouse_is_pressing = False

                if (roi_distance_mode == True) and (restricted_area_redrectangle_roi_pyqt == False) :
                    restricted_area_distancing(people_coords, im0, distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y)

                # Pass detections to deepsort
                if len(xywhs):
                    im0 = deepsort.update(xywhs, confss, im0, wander, fw_queue, fight_time, falling_down_time , smoking_time,action_mode, count_graph, Choose_pyqt_pc)

                # 파이어베이스 이미지 업로드
                if Choose_pyqt_smartphone == True:
                    now = datetime.now()
                    sec = now.second
                    if sec % 4 == 0 and tm2 != sec:
                        cv2.imwrite("capture.png", im0)
                        blob = bucket.blob(fire_filename)
                        blob.upload_from_filename(fire_filename)
                    tm2 = sec

                # db연결
                conn = pymysql.connect(host="localhost", user="root", password="123456789", db="cctv_db",
                                       charset="utf8")
                curs = conn.cursor()
                curs2 = conn.cursor()
                sqltime = "select id, action,time from all_in_one where action is not null"
                sqlroaming = "select id,situation,time " \
                             "from all_in_one where situation is not null"
                curs.execute(sqltime)
                curs2.execute(sqlroaming)
                rows = curs.fetchall()
                rows2 = curs2.fetchall()

                # db박스 초기화 후 데이터 마다 색상 변경 - 위젯에 출력
                if Choose_pyqt_pc == True:
                    widg.clear()
                    widg2.clear()
                    for i in range(0, len(rows)):
                        if (str(rows[i][1]) == 'danger'):
                            widg.setTextColor(QColor(255, 51, 0))
                            widg.append(str(rows[i]))
                        else:
                            widg.setTextColor(QColor(255, 127, 0))
                            widg.append(str(rows[i]))
                    for i in range(0, len(rows2)):
                        if (str(rows2[i][1]) == 'loitering'):
                            widg2.setTextColor(QColor(139, 0, 255))
                            widg2.append(str(rows2[i]))
                        else:
                            widg2.setTextColor(QColor(0, 255, 0))
                            widg2.append(str(rows2[i]))
            else:
                deepsort.increment_ages()

            # Print time (inference + NMS), 프레임 밀리는 오류 예외처리를 통해 넘어감
            try:
                runtime_fps = 1 / (time.time() - start)
            except:
                runtime_fps = 100
                pass

            pbar.set_description(f"runtime_fps: {runtime_fps}")
            pbar.update(1)

            # Stream results
            height, width = im0.shape[:2]

            if view_img:
                # 직사각형 메시지 선택 했을때 OR 폴리곤 메시지 선택 했을때만 실행
                if Choose_pyqt_Rect == True or Choose_pyqt_Polygon == True :
                    # PYQT ROI 활성화 버튼을 누를시 ROI Settings 윈도우창 생성 및 직사각형 그리기
                    if redrectangle_roi_pyqt == True:
                        cv2.namedWindow("ROI Settings")
                        cv2.setMouseCallback("ROI Settings", Mouse_Callback_Rect_One)
                        im0, start_x, end_x, start_y, end_y = draw_roi_rectangle(im0, step, start_x, end_x, start_y, end_y)
                        # 마우스 눌렀다 때면 PYQT 창에도 휘발성으로 노란색 직사각형을 그림
                        if step == 3 :
                            #redrectangle_roi_pyqt = False # 만약 눌렀다 떈 동시에 윈도우창을 끄고 싶으면
                            cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
                        cv2.imshow("ROI Settings", im0)
                        key = cv2.waitKey(1)
                        # esc 누를경우, ROI 직사각형 좌표 설정 종료 및 RoI Mode 활성화
                        if key == 27:
                            redrectangle_roi_pyqt = False
                            cv2.destroyWindow("ROI Settings")
                            roi_mode_on = True
                            mouse_is_pressing = False
                    # PYQT ROI 활성화 버튼을 누를시 ROI Settings 윈도우창 생성 및 폴리곤 점 찍고 선 이어주기
                    elif redpolygon_roi_pyqt == True:
                            cv2.namedWindow("Polygon_Window")
                            cv2.resizeWindow(winname='Polygon_Window', width=1280, height=960)
                            cv2.setMouseCallback("Polygon_Window", Mouse_Callback_Polygon)
                            im0 = draw_roi_polygon(im0, step, polygon_xy_list)
                            cv2.imshow("Polygon_Window", im0)
                            key = cv2.waitKey(1)
                            # esc 누를경우, ROI 직사각형 좌표 설정 종료 및 RoI Mode 활성화
                            if key == 27:
                                np_xy = np.array(polygon_xy_list)
                                p_x, p_y, p_w, p_h  = cv2.boundingRect(np_xy)
                                start_x, end_x, start_y, end_y = p_x, (p_w + p_x), p_y, (p_h + p_y)
                                redpolygon_roi_pyqt = False
                                cv2.destroyWindow("Polygon_Window")
                                roi_mode_on = True

                # ROI Mode 활성화 시 PYQT 창에 직사각형 고정, 이부분 오류날수도
                if roi_mode_on == True:
                    # 직사각형 메시지 선택 했을때
                    if Choose_pyqt_Rect == True :
                        cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
                    elif Choose_pyqt_Polygon == True :
                        np_xy = np.array(polygon_xy_list)
                        for i in range(len(np_xy)):
                            cv2.polylines(im0, [np_xy], True, (0, 0, 255), 3)

                writer.write(im0)

                # 파이큐티 화면 출력 VideoSignal1
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                h, w, c = im0.shape
                qImg = QtGui.QImage(im0.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                VideoSignal1.setPixmap(pixmap)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            vid_writer.write(im0.astype(np.uint8))
            im0 = Image.fromarray(im0[..., ::-1])
            print(im0)
            # im0.save(pipe.stdin, 'JPEG')
            print("반복문 끝")

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        vid_writer.release()
        pipe.stdin.close()
        pipe.wait()
        pbar.close()
    writer.release()

    print('Done. (%.3fs)' % (time.time() - t0))


# PYQT 버튼 동작 함수
def image_to_data(image):
    pb = np.array(image.convert('RGB')).astype('uint16')
    color = ((pb[:,:,0] & 0xF8) << 8) | ((pb[:,:,1] & 0xFC) << 3) | (pb[:,:,2] >> 3)
    return np.dstack(((color >> 8) & 0xFF, color & 0xFF)).flatten().tolist()

def start():
    global args
    conn = pymysql.connect(host="localhost", user="root", password="123456789", db="cctv_db",
                           charset="utf8")
    curs = conn.cursor()
    sql = """delete from all_in_one where no !=1"""
    curs.execute(sql)
    conn.commit()
    print("started..")
    detect(args)

def stop():
    global running
    print("stoped..")
    running = False

def roi_on():
    global Choose_pyqt_Rect, Choose_pyqt_Polygon
    global redrectangle_roi_pyqt, redpolygon_roi_pyqt
    print("start roi..")
    if Choose_pyqt_Rect == True :
        redrectangle_roi_pyqt = True
        redpolygon_roi_pyqt = False
    elif Choose_pyqt_Polygon == True :
        redrectangle_roi_pyqt = False
        redpolygon_roi_pyqt = True

def roi_off():
    global start_x, start_y, end_x, end_y, step, roi_mode_on, mouse_is_pressing, redrectangle_roi_pyqt, redpolygon_roi_pyqt
    print("roi off..")
    redrectangle_roi_pyqt = False
    redpolygon_roi_pyqt = False
    step = 0
    start_x, start_y, end_x, end_y = 0, 0, 0, 0
    roi_mode_on = False
    mouse_is_pressing = False
    cv2.destroyWindow("ROI Settings")
    polygon_xy_list.clear()
    cv2.destroyWindow("Polygon_Window")

def onExit():
    print("exit")
    stop()
    sys.exit()

def connecttion():
    global Choose_pyqt_Rect, Choose_pyqt_Polygon
    if radio_rectangle.isChecked():
        Choose_pyqt_Rect = True
        Choose_pyqt_Polygon = False
        print("You choose Rect_Mode ..")

    elif radio_polygon.isChecked():
        Choose_pyqt_Rect = False
        Choose_pyqt_Polygon = True
        print("You choose Polygon_Mode ..")
    else :
        Choose_pyqt_Rect = False
        Choose_pyqt_Polygon = False
        print("Please select ROI Mode")

def mode_fight():
    global action_mode
    action_mode = "fight"
def mode_falling_down():
    global action_mode
    action_mode = "falling_down"
def mode_smoking():
    global action_mode
    action_mode = "smoking"
def mode_disable():
    global action_mode
    action_mode = "disable"


def mode_distance():
    global distance_mode
    distance_mode = True
def mode_ndistance():
    global distance_mode
    distance_mode = False

def mode_restricted():
    global restricted_area_redrectangle_roi_pyqt
    restricted_area_redrectangle_roi_pyqt = True
def mode_nrestricted():
    global roi_distance_mode, restricted_area_redrectangle_roi_pyqt, distance_roi_step, distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y, mouse_is_pressing
    roi_distance_mode = False
    restricted_area_redrectangle_roi_pyqt = False
    distance_roi_st_x, distance_roi_st_y, distance_roi_end_x, distance_roi_end_y = 0,0,0,0
    mouse_is_pressing = False
    distance_roi_step = 0
    cv2.destroyWindow("Restricted Area ROI Settings")

def mode_save_detection():
    global safety_mode_on
    safety_mode_on = True
def mode_save_ndetection():
    global safety_mode_on
    safety_mode_on = False

def device_han():
    global Choose_pyqt_pc, Choose_pyqt_smartphone, Choose_pyqt_disable
    if radio_pc.isChecked():
        Choose_pyqt_pc = True
        Choose_pyqt_smartphone = False
        Choose_pyqt_disable = False
        print("You choose PC_Device ..")
    elif radio_smartphone.isChecked():
        Choose_pyqt_pc = False
        Choose_pyqt_smartphone = True
        Choose_pyqt_disable = False
        print("You choose SmartPhone_Device ..")
    elif radio_disable.isChecked():
        Choose_pyqt_pc = False
        Choose_pyqt_smartphone = False
        Choose_pyqt_disable = True
        print("You choose Disable_Device ..")
    else :
        Choose_pyqt_pc = False
        Choose_pyqt_smartphone = False
        Choose_pyqt_disable = True
        print("Please select Device")


# 그래프 데이터 가져오기
def get_data():
    global fw_queue, action_mode, count_graph, distance_graph, restricted_distance_graph, safety_distance_graph, tm
    global distance_mode, roi_distance_mode, restricted_area_redrectangle_roi_pyqt, safety_mode_on

    new_time_data = int(time.time())
    now = datetime.now()
    sec = now.second

    # 인원 수 그래프 출력
    if not count_graph:
        y1 = 0
    else :
        y1 = count_graph[-1]
    #파이어베이스
    if Choose_pyqt_smartphone == True :
        if sec % 4 == 0 and tm != sec:
            print(sec)
            ref = db.reference('인원수')  # 경로가 없으면 생성한다.
            ref.update({'인원수': str(y1)})

    # 객체 간 거리 측정, 제한구역, 안전복 위험도 출력
    y2 = -0.1
    y2_2 = 0
    if not (distance_graph + restricted_distance_graph + safety_distance_graph):
        y2 = -0.1
        y2_2 = 0
    else:
        if distance_graph:
            y2 += distance_graph[-2]
            y2_2 += distance_graph[-1]
        if restricted_distance_graph:
            y2 += restricted_distance_graph[-2]
            y2_2 += restricted_distance_graph[-1]
        if safety_distance_graph:
            y2 += safety_distance_graph[-2]
            y2_2 += safety_distance_graph[-1]
    # 파이어베이스 접근 범위 거리
    if Choose_pyqt_smartphone == True:
        if (roi_distance_mode == True) and (restricted_area_redrectangle_roi_pyqt == False):
            if sec % 4 == 0 and tm != sec :
                ref = db.reference('제한구역')  # 경로가 없으면 생성한다.
                ref.update({'경고': str(y2+y2_2+0.1)})
        # 파이어베이스 객체 간 거리
        if distance_mode == True:
            if sec % 4 == 0 and tm != sec :
                ref = db.reference('충돌사고')  # 경로가 없으면 생성한다.
                ref.update({'경고': str(y2 + y2_2 + 0.1)})
        # 파이어베이스 안전복
        if safety_mode_on == True:
            if sec % 4 == 0 and tm != sec :
                ref = db.reference('안전보호구')  # 경로가 없으면 생성한다.
                ref.update({'경고': str(y2+y2_2+0.1)})

    # 싸움, 쓰러짐 점수 그래프 출력
    if not fw_queue:
        y3 = 0
    else :
        graph_score = 0
        for i in range(len(fw_queue)):
            graph_score = graph_score + fw_queue[i]
        y3  = graph_score
    # 파이어베이스 폭력 및 쓰러짐
    if Choose_pyqt_smartphone == True:
        if action_mode == "fight" :
            if sec % 4 == 0 and tm != sec :
                ref = db.reference('폭력감지')  # 경로가 없으면 생성한다.
                ref.update({'경고': str(y3)})

        if action_mode == "falling_down":
            if sec % 4 == 0 and tm != sec:
                ref = db.reference('쓰러짐감지')  # 경로가 없으면 생성한다.
                ref.update({'경고': str(y3)})

    win2.update_plot(new_time_data, y1, y2, y2_2, y3)
    distance_graph.clear()
    restricted_distance_graph.clear()
    safety_distance_graph.clear()
    count_graph.clear()
    tm = sec

#  웹캠 또는 영상으로 지정하는 변수 파이큐티 사용 하기위해
device = '0'
#device = 'inference/a_Slomo.mp4'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        #default='yolov5/weights/yolov5s.pt', help='model.pt path')
                        default='yolov5/weights/ppe_yolo_n.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=device, help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    # MP4파일을 이용하여 실행할때
    if device != '0' :
        detect(args)

    # PTQT 디자인 및 위젯 생성
    else:
        app = QtWidgets.QApplication(sys.argv)
        win = QtWidgets.QWidget()

        win2 = Realime_graph()
        # 실시간 그래프 불러오기
        mytimer = QTimer()
        # 1초마다 갱신
        mytimer.start(1000)
        mytimer.timeout.connect(get_data)
        win2.show()

        vbox = QtWidgets.QHBoxLayout()
        vbox2 = QtWidgets.QVBoxLayout()
        vbox3 = QtWidgets.QVBoxLayout()

        vbox4 = QtWidgets.QVBoxLayout()
        vbox5 = QtWidgets.QHBoxLayout()
        vbox6 = QtWidgets.QVBoxLayout()
        vbox7 = QtWidgets.QHBoxLayout()
        vbox8 = QtWidgets.QVBoxLayout()
        vbox9 = QtWidgets.QVBoxLayout()
        vbox10 = QtWidgets.QVBoxLayout()
        vbox11 = QtWidgets.QVBoxLayout()
        vbox12 = QtWidgets.QVBoxLayout()

        vbox15 = QtWidgets.QHBoxLayout()

        vbox13 = QtWidgets.QVBoxLayout()
        vbox14 = QtWidgets.QVBoxLayout()

        # Camera on/off
        gbox = QtWidgets.QGroupBox()
        gbox.setTitle("Camera")
        gbox.setLayout(vbox3)
        btn_start = QtWidgets.QPushButton("Camera ON")
        btn_stop = QtWidgets.QPushButton("Camera OFF")
        vbox3.addWidget(btn_start)
        vbox3.addWidget(btn_stop)

        # ROI Mode Type
        gbox3 = QtWidgets.QGroupBox()
        gbox3.setTitle("ROI Mode Type")
        gbox3.setLayout(vbox5)
        radio_polygon = QtWidgets.QRadioButton("Polygon")
        radio_rectangle = QtWidgets.QRadioButton("Rectangle")
        vbox5.addWidget(radio_polygon)
        vbox5.addWidget(radio_rectangle)

        # ROI Mode on/off
        gbox2 = QtWidgets.QGroupBox()
        gbox2.setTitle("ROI Mode")
        gbox2.setLayout(vbox4)
        btn_roi_on = QtWidgets.QPushButton("ROI Mode On")
        btn_roi_off = QtWidgets.QPushButton("ROI Mode Off")
        vbox4.addWidget(btn_roi_on)
        vbox4.addWidget(btn_roi_off)

        # Actioon Detection
        gbox4 = QtWidgets.QGroupBox()
        gbox4.setTitle("Action Detection")
        gbox4.setLayout(vbox6)
        btn_fight = QtWidgets.QPushButton("Fight Detection Mode")
        btn_falling_down = QtWidgets.QPushButton("Falling Down Detection Mode")
        btn_smoking = QtWidgets.QPushButton("Smoking Detection Mode")
        btn_disable = QtWidgets.QPushButton("Disable All Detections")
        vbox6.addWidget(btn_fight)
        vbox6.addWidget(btn_falling_down)
        vbox6.addWidget(btn_smoking)
        vbox6.addWidget(btn_disable)

        # Distance Detection
        gbox10 = QtWidgets.QGroupBox()
        gbox10.setTitle("Distance Detection")
        gbox10.setLayout(vbox12)
        # Collision Risk Detection Mode
        gbox8 = QtWidgets.QGroupBox()
        gbox8.setTitle("Collision Risk Detection Mode")
        gbox8.setLayout(vbox10)
        btn_distance = QtWidgets.QPushButton("Collision Risk ON")
        btn_nditance = QtWidgets.QPushButton("Collision Risk OFF")
        vbox10.addWidget(btn_distance)
        vbox10.addWidget(btn_nditance)
        # Restricted Area Detection Mode
        gbox9 = QtWidgets.QGroupBox()
        gbox9.setTitle("Restricted Area Detection Mode")
        gbox9.setLayout(vbox11)
        btn_restricted = QtWidgets.QPushButton("Restricted Area ON")
        btn_nrestricted = QtWidgets.QPushButton("Restricted Area OFF")
        vbox11.addWidget(btn_restricted)
        vbox11.addWidget(btn_nrestricted)

        vbox12.addWidget(gbox8)
        vbox12.addWidget(gbox9)

        # Object Detection
        gbox12 = QtWidgets.QGroupBox()
        gbox12.setTitle("Object Detection")
        gbox12.setLayout(vbox14)
        # Safety Gear Detection Mode
        gbox11 = QtWidgets.QGroupBox()
        gbox11.setTitle("Safety Gear Detection Mode")
        gbox11.setLayout(vbox13)
        btn_save = QtWidgets.QPushButton("Safety Gear ON")
        btn_nsave = QtWidgets.QPushButton("Safety Gear OFF")
        vbox13.addWidget(btn_save)
        vbox13.addWidget(btn_nsave)

        vbox14.addWidget(gbox11)

        # Database on/off
        gbox13 = QtWidgets.QGroupBox()
        gbox13.setTitle("Information Output Device") # PC, Smart Phone, Disable
        gbox13.setLayout(vbox15)
        radio_pc = QtWidgets.QRadioButton("PC")
        radio_smartphone = QtWidgets.QRadioButton("Smart Phone")
        radio_disable = QtWidgets.QRadioButton("Disable All Device")
        vbox15.addWidget(radio_pc)
        vbox15.addWidget(radio_smartphone)
        vbox15.addWidget(radio_disable)

        # Information
        gbox5 = QtWidgets.QGroupBox()
        gbox5.setTitle("Information")
        # Abnormal Action
        gbox6 = QtWidgets.QGroupBox()
        gbox6.setTitle("Abnormal Action")
        widg = QtWidgets.QTextEdit()
        vbox8.addWidget(widg)
        # Loitering People
        gbox7 = QtWidgets.QGroupBox()
        gbox7.setTitle("Loitering People")
        widg2 = QtWidgets.QTextEdit()
        vbox9.addWidget(widg2)

        gbox6.setLayout(vbox8)
        gbox7.setLayout(vbox9)
        vbox7.addWidget(gbox6)
        vbox7.addWidget(gbox7)

        gbox5.setLayout(vbox7)

        vbox2.addWidget(gbox)
        vbox2.addWidget(gbox3)
        vbox2.addWidget(gbox2)
        vbox2.addWidget(gbox4)
        vbox2.addWidget(gbox10)
        vbox2.addWidget(gbox12)
        vbox2.addWidget(gbox13)
        vbox2.addWidget(gbox5)

        win.setStyleSheet(
            "background-color: rgb(34, 32, 41)"
        )
        gbox.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox2.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox3.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox4.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox5.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox8.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox9.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox10.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox11.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox12.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )
        gbox13.setStyleSheet(
            "color: white;"
            "background-color: rgb(47, 42, 53)"
        )

        VideoSignal1 = QtWidgets.QLabel()

        win.setWindowTitle("Prison Artificial Intelligent CCTV")
        win.resize(50, 100)

        btn_start.clicked.connect(start)
        btn_stop.clicked.connect(stop)

        btn_roi_on.clicked.connect(roi_on)
        btn_roi_off.clicked.connect(roi_off)

        radio_polygon.clicked.connect(connecttion)
        radio_rectangle.clicked.connect(connecttion)

        btn_fight.clicked.connect(mode_fight)
        btn_falling_down.clicked.connect(mode_falling_down)
        btn_smoking.clicked.connect(mode_smoking)
        btn_disable.clicked.connect(mode_disable)

        btn_distance.clicked.connect(mode_distance)
        btn_nditance.clicked.connect(mode_ndistance)

        btn_restricted.clicked.connect(mode_restricted)
        btn_nrestricted.clicked.connect(mode_nrestricted)

        btn_save.clicked.connect(mode_save_detection)
        btn_nsave.clicked.connect(mode_save_ndetection)

        radio_pc.clicked.connect(device_han)
        radio_smartphone.clicked.connect(device_han)
        radio_disable.clicked.connect(device_han)

        vbox.addWidget(VideoSignal1)
        vbox.addLayout(vbox2)

        win.setLayout(vbox)
        win.show()
        sys.exit(app.exec_())
