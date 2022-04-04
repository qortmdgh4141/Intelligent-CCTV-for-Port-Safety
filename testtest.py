def Mouse_Callback(event, x, y, flags, step, start_x, end_x, start_y, end_y, mouse_is_pressing):
    # Press The Left Button
    if event == cv2.EVENT_LBUTTONDOWN:
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

    return step, start_x, end_x, start_y, end_y, mouse_is_pressing

def draw_roi_rectangle(img, step, start_x, end_x, start_y, end_y):

    if step == 1:
        cv2.circle(img, (start_x, start_y), 10, (0, 255, 0), -1)
    # Moving The Mouse
    elif step == 2:
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    # Release Of The Mouse
    elif step == 3:
        # If Start X Position Is Bigger Than End X
        if start_x > end_x and start_y < end_y:  # 오른쪽 위에서 왼쪽 아래로 드래그 할 시\
            if end_x < 0:
                end_x = 0
            start_x, end_x = end_x, start_x
        elif start_x > end_x and start_y > end_y:  # 오른쪽 아래서 왼쪽 위로 드래그 할 시
            if end_y < 0:
                end_y = 0
            start_y, end_y = end_y, start_y
            print(start_y)
            start_x, end_x = end_x, start_x
        elif start_x < end_x and start_y > end_y:  # 왼쪽 아래서 오른쪽 위로 드래그시
            if end_y < 0:
                end_y = 0
            start_y, end_y = end_y, start_y

        return img, step, start_x, end_x, start_y, end_y