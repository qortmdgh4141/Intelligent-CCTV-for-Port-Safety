import torch
# GPU가 사용 가능한지 확인합니다.
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import mxnet as mx
# MX GPU가 사용 가능한지 확인합니다.
print(mx.context.num_gpus())
with mx.Context('gpu', 0):
    print(mx.current_context())

def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)

if not gpu_device():
    print('No GPU device found!')
