import os
import socket
import struct
import sys
import threading
import time

from PIL import Image, ImageGrab
from skimage import io

import argparse

import shutil

from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized





def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('192.168.1.6', 9999))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    

    print ("Waiting...")

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=detect, args=(conn, addr))
        t.start()


def detect(conn,addr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\trainer\Desktop\cut_picture/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # if os.path.exists(out):  # output dir
    #     shutil.rmtree(out)  # delete dir
    # os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    print ('Accept new connection from {0}'.format(addr))
    while 1:
        fileinfo_size = struct.calcsize('128sl')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.strip(str.encode('\00'))
            filepath = r'C:\Users\trainer\Desktop\cut_picture/'
            new_filename = os.path.join(str.encode(filepath), str.encode('new_') + fn)
            recvd_size = 0  
            fp = open(new_filename, 'wb')
            while not recvd_size == filesize:
                if filesize - recvd_size > 10240:
                    data = conn.recv(10240)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            dataset = LoadImages(source, img_size=imgsz)
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = model(img, augment=opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    # if webcam:  # batch_size >= 1
                    #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    # else:
                    p, s, im0 = path, '', im0s

                    # save_path = str(Path(out) / Path(p).name)
                    # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                    # s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        # Write results
                        dt_boxes, rec_res = [],[] 
                        for *xyxy, conf, cls in reversed(det):
                        #     if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                            res = ('%g ' * len(line)) % line
                            res = res.split()
                            rec_res.append(names[int(cls)])
                            dt_boxes.append((res[1],res[2],res[3],res[4]))
                        send_data = (dt_boxes, rec_res)
                        conn.sendall(str(send_data).encode())
                        try:
                            os.remove(new_filename)
                        except PermissionError as e:
                            print(e)
                            os.remove(new_filename)



if __name__ == '__main__':
    socket_service()





















