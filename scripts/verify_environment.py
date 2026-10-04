"""Offline smoke checks; does not connect to cameras or send robot actions."""
import argparse
import importlib
import importlib.metadata as metadata
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'orcalab':
        raise RuntimeError('Activate orcalab or use conda run -n orcalab')
    print('Python:', sys.executable)
    for name in ('orca-lab', 'orca-gym'):
        if metadata.version(name) != '26.8.2':
            raise RuntimeError(f'{name} must be 26.8.2')
    for name in ('orcalab.launcher', 'orca_gym.environment',
                 'orca_gym.sensor.rgbd_camera', 'websockets', 'yaml', 'ultralytics'):
        importlib.import_module(name)
    import av
    import cv2
    import numpy as np
    import torch
    import torchvision

    # Exercise image encode/decode and a feature used for panel pose estimation.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', frame)
    if not ok or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), frame):
        raise RuntimeError('OpenCV image round trip failed')
    points = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    matrix = cv2.getPerspectiveTransform(points, points * 10)
    if not np.allclose(cv2.perspectiveTransform(points[None], matrix)[0], points * 10):
        raise RuntimeError('OpenCV geometry check failed')
    av.codec.Codec('h264', 'r')
    # NMS catches incompatible torch/torchvision binary builds.
    indices = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
                                  torch.tensor([0.9, 0.8]), 0.5)
    if indices.tolist() != [0]:
        raise RuntimeError('Torchvision NMS failed')
    cuda = torch.cuda.is_available()
    print('Torch:', torch.__version__, 'CUDA:', cuda)
    if args.require_cuda and not cuda:
        raise RuntimeError('CUDA requested but unavailable; inspect nvidia-smi and driver')
    if cuda:
        device = torch.cuda.get_device_name(0)
        value = (torch.ones(2, device='cuda') * 2).sum().item()
        if value != 4:
            raise RuntimeError('CUDA computation failed')
        print('GPU:', device)
    print('Environment verification OK (imports/image/geometry/H.264 decoder/NMS).')
    print('OrcaLab launch, real camera streams, NVENC and robot control still need local validation.')


if __name__ == '__main__':
    main()
