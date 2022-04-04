from collections import deque
import time
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
from subprocess import Popen, PIPE
from PIL import Image
# from gluoncv.utils.filesystem import try_import_decord

SAMPLE_DURATION = 32
frames = deque(maxlen=SAMPLE_DURATION)


video_file = "smoking3people.mov"


# load video
vid = cv2.VideoCapture(video_file)
fps = vid.get(cv2.CAP_PROP_FPS)
model_name = 'i3d_resnet50_v1_hmdb51'
net = get_model(model_name, pretrained=True)

with mx.Context('gpu', 0):  # Context changed in `with` block.
    net.collect_params().reset_ctx(ctx=mx.current_context())

save_path = f"inference/output/{video_file}_full_action.mp4"
pipe = Popen([
        'ffmpeg', '-loglevel', 'quiet', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', f'{fps}', 
        '-i', '-', '-vcodec', 'libx264', '-crf', '28', '-preset', 'veryslow', '-framerate', f'{fps}', f'{save_path}'
    ], stdin=PIPE)

while True:
    start = time.time()
    ret, frame = vid.read()
    if not ret:
        break
    frames.append(frame)
    # print([f.shape for f in nframes])
    
    # cv2.imshow("frame", frame)
    if len(frames) < SAMPLE_DURATION:
        continue

    # if len(clip_input) == SAMPLE_DURATION:
    clip_input = frames
    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    pred = net(nd.array(clip_input))
    classes = net.classes
    topK = 1
    ind = nd.topk(pred, k=topK)[0].astype('int')
    # print('The input video clip is classified to be')
    try:
        runtime_fps = 1 / (time.time() - start)
    except ZeroDivisionError:
        runtime_fps = 0

    action = classes[ind[0].asscalar()]
    label = f"{runtime_fps:.2f}_{action}"
    cv2.putText(frame, label, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    frame = Image.fromarray(frame[..., ::-1])
    print(f"INFO: FPS {runtime_fps:.2f} {action}")
    frame.save(pipe.stdin, 'JPEG')
    

pipe.stdin.close()
pipe.wait()