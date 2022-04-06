import sys

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QPushButton, QWidget, QApplication, QComboBox

sys.path.insert(0, './yolov5')
from subprocess import Popen, PIPE
from PIL import Image
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
import os

# PYQT 카메라 ON,OFF 버튼 선택
running = True

# PYQT 직사각형, 폴리곤 ROI 버튼 선택
redrectangle_roi_pyqt = False
redpolygon_roi_pyqt = False

Choose_pyqt_Rect = False
Choose_pyqt_Polygon = False

#Color 윈도우창 ESC버튼 누른 순간 ROI 모드 활성화
roi_mode_on = False

# 마우스 상태 및 직사각형 ROI 좌표 초기화,
mouse_is_pressing, step = False, 0
start_x, start_y, end_x, end_y = 0,0,0,0
polygon_xy_list = []

# 직사각형 ROI 마우스 이벤트 핸들러 함수, 좌푯값 저장
def Mouse_Callback_Rect(event, x, y, flags, params):
    # Press The Left Button
    global step , start_x, end_x, start_y, end_y, mouse_is_pressing

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
        print("Error : Mouse_Callback_Rect 함수 예외처리")

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

# 폴리곤 ROI 그리기 및 좌표값 변환 함수
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
    global running
    global redrectangle_roi_pyqt, redpolygon_roi_pyqt, roi_mode_on
    global start_x, start_y, end_x, end_y, polygon_xy_list
    global step, mouse_is_pressing
    global Choose_pyqt_Rect, Choose_pyqt_Polygon


    # pyqt start 버튼 누르면 다시 실행 될 수 있도록 True 설정
    running =True

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

        # Color 윈도우창을 이용해 마우스 좌푯값을 수정했을때만 실행
        if not (start_x == 0 or end_x == 0 or start_y == 0 or end_y == 0):
            # 직사각형, 폴리곤 메시지박스 선택 했을때
            if Choose_pyqt_Rect == True or Choose_pyqt_Polygon == True:
                # ROI Mode 활성화 및 마우스 좌표 설정이 끝났을 때
                if (roi_mode_on == True) and (mouse_is_pressing==False) :
                    end_y = end_y - ((end_y - start_y) % 32)
                    end_x = end_x - ((end_x - start_x) % 32)

                    # 실제, 직사각형 ROI 영역 지정 ㅎㅎ
                    img = img[:, :, start_y: end_y, start_x: end_x]
                    print(f"직사각형 ROI 영역 좌푯값 == start_x : {start_x}, start_y : {start_y}, end_x : {end_x}, end_y : {end_y}")
            else:
                print("PYQT 메시지 박스 '직사각형' 과 '폴리곤' 중 선택 하세요")


        img = torch.from_numpy(img).to(device)
        #print(f'0번째 -->  {img.shape}')
        img = img.half() if half else img.float()  # uint8 to fp16/32
        #print(f'1번째 -->  {img.shape}')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #print(f'2번째 -->  {img.shape}')
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            #print(f'3번째 -->  {img.shape}')
        #print(f'3번째 -->  {img.shape}')

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #print(f'4번째 -->  {img.shape}')
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, im0 = path, im0s

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

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                im0 = deepsort.update(xywhs, confss, im0, wander)

                # # draw boxes for visualization
                # if len(outputs) > 0:
                #     bbox_xyxy = outputs[:, :4]
                #     identities = outputs[:, -1]
                #     draw_boxes(im0, bbox_xyxy, identities)
            else:
                deepsort.increment_ages()


            # Print time (inference + NMS), 프레임 밀리는 오류 예외처리를 통해 넘어감
            try:
                runtime_fps = 1 / (time.time() - start)
            except:
                runtime_fps = 100
                pass

            # print(f"Runtime FPS: {runtime_fps:.2f}")
            pbar.set_description(f"runtime_fps: {runtime_fps}")
            pbar.update(1)
            # Stream results
            height, width = im0.shape[:2]

            if view_img:
                #cv2.imshow(p, im0)

                # 직사각형 메시지 선택 했을때 OR 폴리곤 메시지 선택 했을때만 실행
                if Choose_pyqt_Rect == True or Choose_pyqt_Polygon == True :
                    # PYQT ROI 활성화 버튼을 누를시 Color 윈도우창 생성 및 직사각형 그리기
                    if redrectangle_roi_pyqt == True:

                        cv2.namedWindow("Color")
                        cv2.setMouseCallback("Color", Mouse_Callback_Rect)

                        im0, start_x, end_x, start_y, end_y = draw_roi_rectangle(im0, step, start_x, end_x, start_y, end_y)

                        # 마우스 눌렀다 때면 PYQT 창에도 휘발성으로 노란색 직사각형을 그림
                        if step == 3 :
                            #redrectangle_roi_pyqt = False # 만약 눌렀다 떈 동시에 윈도우창을 끄고 싶으면
                            cv2.rectangle(im0, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)

                        cv2.imshow("Color", im0)
                        key = cv2.waitKey(1)

                        # esc 누를경우, ROI 직사각형 좌표 설정 종료 및 RoI Mode 활성화
                        if key == 27:
                            redrectangle_roi_pyqt = False
                            cv2.destroyWindow("Color")
                            roi_mode_on = True
                            mouse_is_pressing = False

                    # PYQT ROI 활성화 버튼을 누를시 Color 윈도우창 생성 및 폴리곤 점 찍고 선 이어주기
                    elif redpolygon_roi_pyqt == True:

                            cv2.namedWindow("Polygon_Window")
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

                            # 테스트 박스
                            pp_x, pp_y, pp_w, pp_h = cv2.boundingRect(np_xy)
                            cv2.rectangle(im0, (pp_x, pp_y), ((pp_w + p_x), (pp_h + p_y)), (255, 0, 255), 3)




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

    print('Done. (%.3fs)' % (time.time() - t0))

# PYQT 버튼 동작 함수
def start():
    global args
    print("started..")
    detect(args)

def stop():
    global running
    print("stoped..")
    running = False
    # raise StopIteration

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
    cv2.destroyWindow("Color")
    polygon_xy_list.clear()
    cv2.destroyWindow("Polygon_Window")

def onExit():
    print("exit")
    stop()
    sys.exit()


def connecttion():
    global Choose_pyqt_Rect, Choose_pyqt_Polygon
    if combo_start.currentText() == "Rect":
        Choose_pyqt_Rect = True
        Choose_pyqt_Polygon = False
        print(combo_start.currentText())
        print("You choose Rect_Mode ..")
    elif combo_start.currentText() == "Polygon":
        Choose_pyqt_Rect = False
        Choose_pyqt_Polygon = True
        print(combo_start.currentText())
        print("You choose Polygon_Mode ..")
    elif combo_start.currentText() == "ROI Mode Setting":
        Choose_pyqt_Rect = False
        Choose_pyqt_Polygon = False
        print("Please select ROI Mode")


#  웹캠 또는 영상으로 지정하는 변수 파이큐티 사용 하기위해
device = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=device, help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
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
                        default=[0], help='filter by class')
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
    else :

        app = QtWidgets.QApplication(sys.argv)
        win = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        vbox2 = QtWidgets.QHBoxLayout()
        vbox3 = QtWidgets.QHBoxLayout()
        VideoSignal1 = QtWidgets.QLabel()
        combo_start = QComboBox()
        btn_start = QtWidgets.QPushButton("Camera On")
        btn_stop = QtWidgets.QPushButton("Camera Off")
        btn_roi_on = QtWidgets.QPushButton("ROI 활성화")
        btn_roi_off = QtWidgets.QPushButton("ROI 비활성화")
        win.setWindowTitle("Prison Artificial Intelligent CCTV")
        win.resize(500,200)

        combo_start.addItem("ROI Mode Setting")
        combo_start.addItem("Rect")
        combo_start.addItem("Polygon")
        check = QtWidgets.QPushButton("선택")

        btn_start.clicked.connect(start)
        btn_stop.clicked.connect(stop)
        btn_roi_on.clicked.connect(roi_on)
        btn_roi_off.clicked.connect(roi_off)


        check.clicked.connect(connecttion)
        vbox.addWidget(VideoSignal1)
        vbox.addLayout(vbox2)
        vbox.addLayout(vbox3)

        vbox2.addWidget(btn_start)
        vbox2.addWidget(btn_stop)
        vbox3.addWidget(btn_roi_on)
        vbox3.addWidget(btn_roi_off)
        vbox.addWidget(combo_start)
        vbox.addWidget(check)
        win.setLayout(vbox)
        win.show()
        sys.exit(app.exec_())

    '''
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout()
    VideoSignal1 = QtWidgets.QLabel()
    btn_start = QtWidgets.QPushButton("카메라 켜기")
    btn_stop = QtWidgets.QPushButton("카메라 끄기")
    red_roi = QtWidgets.QPushButton("배회영역 설정")
    vbox.addWidget(VideoSignal1)
    vbox.addWidget(btn_start)
    vbox.addWidget(btn_stop)
    vbox.addWidget(red_roi)
    win.setLayout(vbox)
    win.show()
    btn_start.clicked.connect(start)
    btn_stop.clicked.connect(stop)
    red_roi.clicked.connect(roi)
    app.aboutToQuit.connect(onExit)
    sys.exit(app.exec_())
    '''


