#!/usr/bin/env python

import argparse, random
from facev import DatasetWrapper
from facev.ensemble.img_proc import *
import caffe
import numpy as np
import cv2
from data_loader import Loader
from facev.ensemble.img_proc import flip


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model in caffe')
    parser.add_argument('solver_path', type=str, help='path to solver.txt')
    parser.add_argument('imgs_list_path', type=str, help='path to images list')
    parser.add_argument('labels_path', type=str, help='path to labels')
    parser.add_argument('landmarks_path', type=str, help='path to landmarks')
    parser.add_argument('-f', '--img_fnc', type=str, default="", help='image processing function')
    parser.add_argument('-s', '--solverstate_path', type=str, default='', help='path to snapshot')
    parser.add_argument('-g', '--gpu_num', type=int, default=0, help='gpu_num')
    parser.add_argument('-i', '--iter_num', type=int, default=600000, help='iter_num')
    parser.add_argument('-p', '--processes_num', type=int, default=0, help='processes_num')

    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_num)
    print(' --->>> gpu: %d' % args.gpu_num)

    solver = caffe.SGDSolver(args.solver_path)
    print(' --->>> solver %s loaded' % args.solverstate_path)
    print(' --->>> start iteration: %d' % solver.iter)


    if args.solverstate_path:
        solver.restore(args.solverstate_path)
        print(' --->>> restored from %s' % args.solverstate_path)

    img_fnc = getattr(img_proc, args.img_fnc)
    loader = Loader(solver.net.blobs['data'].data.shape, args.imgs_list_path, args.labels_path, args.landmarks_path,
                    mirror=True, scale=1., img_fnc=img_fnc, processes=args.processes_num)

    while solver.iter < 600000:
        # load input
        cff_data, cff_label = solver.net.blobs['data'].data, solver.net.blobs['label'].data
        imgs, labels = loader.fetch_data()
        cff_data[...] = imgs
        cff_label[...] = labels
        # step
        solver.step(1)
