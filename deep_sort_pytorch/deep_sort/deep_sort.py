import numpy as np
import cv2
import torch
import time
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
from gluoncv.model_zoo import get_model
import mxnet as mx
import pymysql

__all__ = ['DeepSort']
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class DeepSort(object):

    ''' 영일 영상 좌표들
    global ui, io, op, pp
    ui = 144
    io = 1174
    op = 66
    pp = 620'''
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True, action_mode="disable" ):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)

        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        # 싸움 Mode 모델
        model_name = 'i3d_resnet50_v1_sthsthv2'
        self.net = get_model(model_name, pretrained=True)
        with mx.Context('gpu', 0):  # Context changed in `with` block.
            self.net.collect_params().reset_ctx(ctx=mx.current_context())
            # self.net.initialize(force_reinit=True, ctx=mx.current_context())

        # 쓰러짐, 흡연 Mode 모델
        model_name2 = 'i3d_resnet50_v1_hmdb51'
        self.net2 = get_model(model_name2, pretrained=True)
        with mx.Context('gpu', 0):  # Context changed in `with` block.
            self.net2.collect_params().reset_ctx(ctx=mx.current_context())

        # GPU 사용 및 모델 로드 성공인지 출력
        print(f"Currently using {mx.gpu(0)}")
        print('%s model is successfully loaded.' % model_name)

    def update(self, bbox_xywh, confidences, ori_img, wander, fw_queue, fight_time, falling_down_time, smoking_time, action_mode, count_graph, Choose_pyqt_pc):

        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # DB 연결
        conn = pymysql.connect(host="localhost", user='root', password="123456789", db="cctv_db",
                               charset="utf8")
        curs = conn.cursor()
        curs2 = conn.cursor()

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # 인원 수 카운트 변수 초기화
        count = 0

        # 측정 시간 체크
        stime = round(time.time())
        ids = []

        # output bbox identities
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            bbox = self._tlwh_to_xyxy(box)
            # if len(outputs) > 0:
            # bbox_xyxy = outputs[:, :4]
            # identities = outputs[:, -1]
            # update actions
            track.update_frames(bbox, ori_img)
            # print(f"INFO: len action_frames: {len(track.frames)}")

            # track id 보관하는 리스트
            if track.track_id not in ids:
                ids.append(track.track_id)

            x11, y11, x22, y22 = [int(i) for i in bbox]

            # 거리에 따른 라벨 글자 크기 조정
            textsize = (x22 - x11) * 0.02
            if textsize >= 2:
                textsize = 2
            elif textsize <= 1:
                textsize = 1

            # track_id가 존재하고 wander 딕셔너리에 track_id가 없으면 추가
            # wander 딕셔너리에 track_id가 없을 경우
            if track.track_id not in wander:
                wander[track.track_id] = [stime, 0]
            # wander 에 track_id가 있을 경우
            else:
                # 5초 배회하면 배회중
                # 영일 영상 수정
                # if x11 >= 450 and x22 <= 950 and y11 >= 200 and y22 <= 600:
                    if stime - wander[track.track_id][0] >= 10:
                        # num = track.track_id
                        # num = (num)*"  " + str(num) + ","
                        wander_text = f'Pedestrian loitering'
                        # DB 기록 Dangerous 상황, 시간 기록
                        if Choose_pyqt_pc == True:
                            if ((round(time.time()) % 2) == 0):
                                sql = """insert into all_in_one(id, situation, time)
                                                  values(%s, %s, now())"""
                                sql2 = """select distinct id , situation from all_in_one"""
                                curs2.execute(sql2)
                                rows = curs2.fetchall()
                                a = []
                                for i in range(len(rows)):
                                    a.append(rows[i])
                                if (str(track.track_id), 'loitering') not in a:
                                    curs.execute(sql, (track.track_id, 'loitering'))
                                conn.commit()
                        cv2.putText(ori_img, wander_text, (x11, y11 - 5), cv2.FONT_HERSHEY_PLAIN, textsize, [76, 1, 43], 3)
                    else:
                        # DB 기록 Dangerous 상황, 시간 기록
                        if Choose_pyqt_pc == True:
                            if ((round(time.time()) % 2) == 0):
                                sql = """insert into all_in_one(id, situation, time)
                                                values(%s, %s, now())"""
                                sql2 = """select distinct id , situation from all_in_one"""
                                curs2.execute(sql2)
                                rows = curs2.fetchall()
                                a = []
                                for i in range(len(rows)):
                                    a.append(rows[i])
                                if (str(track.track_id), 'intrusion') not in a:
                                    curs.execute(sql, (track.track_id, 'intrusion'))
                                conn.commit()
                        invasion_text = 'intrusion'
                        cv2.putText(ori_img, invasion_text, (x11, y11 - 5), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 255, 0],2)

            # ids 에 저장된 track id 수가 tracker의 개수와 같을 때 차집합을 구해 일정 시간이 지나면 wander의 딕셔너리에서 삭제하여 불필요한 데이터 낭비 최소화
            if len(ids) == len(self.tracker.tracks):
                dif = set(list(wander.keys())) ^ set(ids)
                dif = list(dif)
                print(set(list(wander.keys())) ^ set(ids))
                for i in range(len(dif)):
                    wander[dif[i]][1] += 1
                    if wander[dif[i]][1] >= 10:
                        del wander[dif[i]]

            # action 인식하고 출력하는 부분
            if action_mode == "disable" :
                action = None
            elif action_mode == "fight":
                action = track.get_action(self.net, action_mode,track.track_id, Choose_pyqt_pc)
            elif action_mode == "falling_down":
                action = track.get_action(self.net2, action_mode, Choose_pyqt_pc)
            elif action_mode == "smoking":
                action = track.get_action(self.net2, action_mode, Choose_pyqt_pc)
            else :
                print("action 오류")

            # action이 Warning 일시 점수 부여
            if len(fw_queue) >= 200:
                fw_queue.pop(0)
            if action == 'Warning Action' :
                fw_queue.append(1)
            else:
                fw_queue.append(0)

            # 싸움 상황 일시 초록색, 주황색, 빨간색 단계별
            if action_mode == "fight":
                fw_score = 0
                for i in range(len(fw_queue)):
                    fw_score =  fw_score + fw_queue[i]
                if (fw_score >= 45) :
                    if (fight_time[0] == False):
                        fight_time.clear()
                        fight_time.append(True)
                        fight_time.append(round(time.time()))
                    action = 'Dangerous Action'
                    # DB 기록 Dangerous 상황, 시간 기록
                    if Choose_pyqt_pc == True:
                        sql = """insert into all_in_one(id, action, time)
                                                     values(%s, %s, now())"""
                        sql2 = """select distinct id, action from all_in_one"""
                        curs2.execute(sql2)
                        rows = curs2.fetchall()
                        a = []
                        for i in range(len(rows)):
                            a.append(rows[i])
                        if (str(track.track_id), 'danger') not in a:
                            curs.execute(sql, (track.track_id, 'danger'))
                        conn.commit()

                # Dangerous 상황일 시 빨간색 박스 5초간 지속
                elif fight_time[0] == True:
                    tm_minus = round(time.time())-fight_time[1]
                    if tm_minus > 5:
                        fight_time.clear()
                        fight_time.append(False)
                        fight_time.append(0)
                    else:
                        action = 'Dangerous Action'
                        # DB 기록 Dangerous 상황, 시간 기록
                        if Choose_pyqt_pc == True:
                            sql = """insert into all_in_one(id, action, time)
                                                                                                                                            values(%s, %s, now())"""
                            sql2 = """select distinct id from all_in_one"""
                            curs2.execute(sql2)
                            rows = curs2.fetchall()
                            a = []
                            for i in range(len(rows)):
                                a.append(rows[i][0])
                            if str(track.track_id) not in a:
                                curs.execute(sql, (track.track_id, 'danger'))
                            conn.commit()

            # 쓰러짐 상황 일시 초록색, 주황색, 빨간색 단계별
            elif action_mode == "falling_down":
                fw_score = 0
                for i in range(len(fw_queue)):
                    fw_score =  fw_score + fw_queue[i]
                if (fw_score >= 10) :
                    if (falling_down_time[0] == False):
                        falling_down_time.clear()
                        falling_down_time.append(True)
                        falling_down_time.append(round(time.time()))
                    action = 'Dangerous Action'
                    # DB 기록 Dangerous 상황, 시간 기록
                    if Choose_pyqt_pc == True:
                        sql = """insert into all_in_one(id, action, time)
                                                     values(%s, %s, now())"""
                        sql2 = """select distinct id, action from all_in_one"""
                        curs2.execute(sql2)
                        rows = curs2.fetchall()
                        a = []
                        for i in range(len(rows)):
                            a.append(rows[i])
                        if (str(track.track_id), 'danger') not in a:
                            curs.execute(sql, (track.track_id, 'danger'))
                        conn.commit()
                # Dangerous 상황일 시 빨간색 박스 5초간 지속
                elif falling_down_time[0] == True:
                    tm_minus = round(time.time())-fight_time[1]
                    if tm_minus > 5:
                        falling_down_time.clear()
                        falling_down_time.append(False)
                        falling_down_time.append(0)
                    else:
                        action = 'Dangerous Action'
                        # DB 기록 Dangerous 상황, 시간 기록
                        if Choose_pyqt_pc == True:
                            sql = """insert into all_in_one(id, action, time)
                                                                                                                                        values(%s, %s, now())"""
                            sql2 = """select distinct id from all_in_one"""
                            curs2.execute(sql2)
                            rows = curs2.fetchall()
                            a = []
                            for i in range(len(rows)):
                                a.append(rows[i][0])
                            if str(track.track_id) not in a:
                                curs.execute(sql, (track.track_id, 'danger'))
                            conn.commit()

            # Smoking 상황일 시 파란색 박스 3초간 지속
            elif action_mode == "smoking":
                # 명령 모드 리스트 박스
                smoking_list = ['smoke', 'chew', "eat","shoot_gun","drink","kiss","sit"]
                # key 값은 id, Value값은 시간
                id = int(track.track_id)


                if action in smoking_list:
                    action = 'Smoking Action'
                    smoking_time[id] = round(time.time())

                elif action not in smoking_list:
                    if id in smoking_time:
                        if (round(time.time()) - smoking_time[id]) < 1:
                            action = 'Smoking Action'
                        else:
                            del (smoking_time[id])
                    else:
                        action = ''

            print(f"INFO: action {action}")

            # 영일 영상
            # if x11 >= ui and x22 <= io and y11 >= op and y22 <= pp:

            # 인원 수 추가
            count += 1

            self.draw_boxes(ori_img, bbox, track.track_id, action, action_mode=action_mode)

        # 인원 수 count
        counting_text = "People Counting : {}".format(count)
        cv2.putText(ori_img, counting_text, (10, ori_img.shape[0] - 25), cv2.LINE_AA, 0.85, (0, 0, 0), 2)

        # run 파일에 넘겨줄 인원수 리스트
        count_graph.append(count)

        # 영일 영상 수정
        # cv2.rectangle(ori_img, (ui, op), (io, pp), (0, 0, 255), 3)

        #     track_id = track.track_id
        #     outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        # if len(outputs) > 0:
        #     outputs = np.stack(outputs, axis=0)
        # return outputs

        return ori_img

    """
      TODO:
          Convert bbox from xc_yc_w_h to xtl_ytl_w_h
      Thanks JieChen91@github.com for reporting this bug!
      """

    @staticmethod
    def draw_boxes(img, bbox, identities=None, action=None, offset=(0, 0), action_mode="disable"):

        # for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in bbox]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities)
        color = DeepSort.compute_color_for_labels(id)
        if action:
            label = f"{id} : {action}"
        else:
            label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        # 거리에 따른 라벨 글자 크기 조정
        textsize = (x2 - x1) * 0.02
        if textsize >= 2:
            textsize = 2
        elif textsize <= 1:
            textsize = 1

        # 영일 영상
        # if x1 >= ui and x2 <= io and y1 >= op and y2 <= pp:

        # 초록색,주황색,빨간색 인식 박스 그려주는 부분
        # 싸움 인식 모드
        if action_mode == "fight":
            if action == 'Warning Action' :
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (55, 110, 145, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 127, 255], 2)
            elif action == 'Dangerous Action' :
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (35, 70, 155, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 0, 255], 2)
            else:
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (10, 40, 10, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 255, 0], 2)
        #  쓰러짐 모드
        elif action_mode == "falling_down":
            if action == 'Warning Action':
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (55, 110, 145, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 127, 255], 2)
            elif action == 'Dangerous Action':
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (35, 70, 155, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 0, 255], 2)
            else:
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (10, 40, 10, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 255, 0], 2)

        # smoking 모드
        elif action_mode == "smoking":
            if action == 'Smoking Action' :
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (255, 51, 153, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 0, 255], 2)
            else :
                ROI_box = img[y1: y2, x1: x2]
                ROI_box = cv2.add(ROI_box, (10, 40, 10, 0))
                img[y1: y2, x1: x2] = ROI_box
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 255, 0], 2)

        # 모드 선택 안함
        elif action_mode == "disable" :
            ROI_box = img[y1: y2, x1: x2]
            ROI_box = cv2.add(ROI_box, (10, 40, 10, 0))
            img[y1: y2, x1: x2] = ROI_box
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, textsize, [0, 255, 0], 2)

        return img

    @staticmethod
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
