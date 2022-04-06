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

__all__ = ['DeepSort']
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)


        # Fighting Mode 모델
        model_name = 'i3d_resnet50_v1_sthsthv2'
        # 명령 Mode 모델
        # model_name = 'i3d_resnet50_v1_hmdb51'

        self.net = get_model(model_name, pretrained=True)

        with mx.Context('gpu', 0):  # Context changed in `with` block.
            self.net.collect_params().reset_ctx(ctx=mx.current_context())
            # self.net.initialize(force_reinit=True, ctx=mx.current_context())

        # GPU 사용 및 모델 로드 성공인지 출력
        print(f"Currently using {mx.gpu(0)}")
        print('%s model is successfully loaded.' % model_name)


    def update(self, bbox_xywh, confidences, ori_img, wander):

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

            # track_id가 존재하고 wander 딕셔너리에 track_id가 없으면 추가
            # wander 딕셔너리에 track_id가 없을 경우
            if track.track_id not in wander:
                wander[track.track_id] = [stime, 0]
            # wander 에 track_id가 있을 경우
            else:
                # 5초 배회하면 배회중
                if stime - wander[track.track_id][0] >= 5:
                    # num = track.track_id
                    # num = (num)*"  " + str(num) + ","
                    wander_text = f'Person roaming'
                    # cv2.putText(ori_img, wander_text, (10, ori_img.shape[0] - 50), cv2.LINE_AA, 0.85, (0, 0, 255), 2)
                    x11, y11, x22, y22 = [int(i) for i in bbox]
                    cv2.putText(ori_img, wander_text, (x11, y11 - 5), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 0], 2)
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
            action = track.get_action(self.net)
            #action = 'Hitting something with something'
            print(f"INFO: action {action}")
            count += 1

            self.draw_boxes(ori_img, bbox, track.track_id, action)

            self.draw_boxes_v2(ori_img, bbox, track.track_id)

        # 인원 수 count
        counting_text = "People Counting : {}".format(count)
        cv2.putText(ori_img, counting_text, (10, ori_img.shape[0] - 25), cv2.LINE_AA, 0.85, (0, 0, 255), 2)

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
    def draw_boxes_v2(ori_img, bbox, track_id):
        print("a")

    @staticmethod
    def draw_boxes(img, bbox, identities=None, action=None, offset=(0, 0)):

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
            label = f"{id}_{action}"
        else:
            label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]


        # 초록색 인식 박스 그려주는 부분
        if action != 'Hitting something with something' :
            ROI_box = img[y1: y2, x1: x2]
            ROI_box = cv2.add(ROI_box, (55, 110, 145, 0))
            # 빨간색감
            #ROI_box = cv2.add(ROI_box, (35, 70, 155, 0))
            img[y1: y2, x1: x2] = ROI_box
        else:
            ROI_box = img[y1: y2, x1: x2]
            ROI_box = cv2.add(ROI_box, (10, 40, 10, 0))
            img[y1: y2, x1: x2] = ROI_box

        #cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        #cv2.rectangle(
        #    img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        #cv2.putText(img, label, (x1, y1 +
        #                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        cv2.putText(img, label, (x1 , y1 +
                                 t_size[1] + 4 ), cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)


        # 거리에 따른 라벨 글자 크기 조정
        textsize = (x2 - x1) * 0.02

        if textsize >= 2:
            textsize = 2
        elif textsize <= 1:
            textsize = 1

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
