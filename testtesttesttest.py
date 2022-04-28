def mode_fight():
    global action_mode
    action_mode = "fight"
def mode_control():
    global action_mode
    action_mode = "control"
def mode_disable():
    global action_mode
    action_mode = "disable"

    # action 인식하고 출력하는 부분
    if action_mode == "disable":
        action = None
    elif action_mode == "fight":
        action = track.get_action(self.net)
    elif action_mode == "control":
        action = track.get_action(self.net)
    else:
        print("action 오류")



# Fighting Mode 모델
        model_name = 'i3d_resnet50_v1_sthsthv2'
        # 명령 Mode 모델
        # model_name = 'i3d_resnet50_v1_hmdb51'

        self.net = get_model(model_name, pretrained=True)

        with mx.Context('gpu', 0):  # Context changed in `with` block.
            self.net.collect_params().reset_ctx(ctx=mx.current_context())
            # self.net.initialize(force_reinit=True, ctx=mx.current_context())
