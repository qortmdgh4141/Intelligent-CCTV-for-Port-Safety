if action_mode == "disable":
    action = None
elif action_mode == "fight":
    action = track.get_action(self.net, action_mode)
elif action_mode == "control":
    action = track.get_action(self.net2, action_mode)
else:
    print("action 오류")


# 명령 모드 리스트 박스
        control_list = ['hand on head', 'Get down', 'clap']



ROI_box = cv2.add(ROI_box, (160, 110, 50, 0)) 파랑색
ROI_box = cv2.add(ROI_box, (255, 51, 153, 0)) 보라색