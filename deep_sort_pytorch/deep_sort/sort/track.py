# vim: expandtab:ts=4:sw=4

from collections import deque
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
import numpy as np
import cv2
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """


    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age


        # Fighting Mode 일시 16으로 설정
        self.SAMPLE_DURATION = 16
        # 명령 Mode 일시 32으로 설정

        self.frames = deque(maxlen=self.SAMPLE_DURATION)
        self.action = None


    def update_frames(self, bbox, image):
        # crop image with bbox roi
        # bbox format xmin, ymin, xmax, ymax
        frame = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        frame = cv2.resize(frame, (224, 224))
        self.frames.append(frame)


    def get_action(self, net):
        if len(self.frames) < self.SAMPLE_DURATION:
            return None

        clip_input = self.frames
        transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        clip_input = transform_fn(clip_input)
        print(f"INFO: action input shape:")
        print([clip.shape for clip in clip_input])
        clip_input = np.stack(clip_input, axis=0)


        # Fighting Mode 일시 16으로 설정
        clip_input = clip_input.reshape((-1,) + (16, 3, 224, 224))
        # 명령 Mode 일시 32으로 설정
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))


        # pred는 2차원배열이며, type이 ndarry : <class 'mxnet.ndarray.ndarray.NDArray'>
        pred = net(nd.array(clip_input, ctx=mx.gpu(0)))

        # topk로 라벨 분류 최종 갯수 선택 가능
        classes = net.classes
        topK = 2
        ind = nd.topk(pred, k=topK)[0].astype('int')

        # 라벨 종류 출력
        print("-----------------------------------------------------------------------------------------")
        for i in range(topK):
            print('\t[%s], with probability %.3f.' %
                  (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
        print("-----------------------------------------------------------------------------------------")

        num = ind[0].asscalar()

        # 확률 35% 이상일 시 확률 출력 및 실행
        if round(nd.softmax(pred)[0][ind[0]].asscalar(), 3) >= 0.45:
            print(f'{round(nd.softmax(pred)[0][ind[0]].asscalar(), 2)}')

            return classes[num]

        else:
            return None


    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        return ret


    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]

        return ret


    def increment_age(self):
        self.age += 1
        self.time_since_update += 1


    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()


    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed


    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted


    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative


    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed


    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
