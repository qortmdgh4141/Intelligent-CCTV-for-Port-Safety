import cv2
step = 0
polygon_list = [[]]
def Mouse_Callback_Polygon(event, x, y, flags, params):
    # Press The Left Button
    global step, polygon_list

    if event == cv2.EVENT_LBUTTONDOWN :
        step = 100
        polygon_list.append([x,y])
        #mouse_is_pressing = True


cap = cv2.VideoCapture(0)

while(True):
    print(polygon_list)
    ret, frame = cap.read()    # Read 결과와 frame
    cv2.namedWindow("Color")
    a = cv2.setMouseCallback("Color", Mouse_Callback_Polygon)
    print(a)

    cv2.imshow("Color", frame)
    key = cv2.waitKey(1)

    # esc 누를경우, ROI 직사각형 좌표 설정 종료 및 RoI Mode 활성화
    if key == 27:
        polygon_list.clear()
        print(polygon_list)
        #cv2.destroyWindow("Color")