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