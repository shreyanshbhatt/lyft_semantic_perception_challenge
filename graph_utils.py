import sys, skvideo.io, json, base64
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import argparse
import sys
import time
import helper
import scipy.misc
import timeit
import sklearn
import sklearn.metrics
import cv2
#import sklearn.metrics.f1_score

from scipy.misc import imread, imresize
from glob import glob

file = sys.argv[-1]
FLAGS = None
image_shape = (320, 384)

def load_graph(graph_file):
    config = tf.ConfigProto()
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops

def benchmark(sess, video, binary=True):
    answer_key = {}
    frame = 1
    g = sess.graph
    x = g.get_tensor_by_name('image_input:0')
    keep_prob = g.get_tensor_by_name('keep_prob:0')
    out = g.get_tensor_by_name('prediction_softmax:0')
    output_compare = []
    times = []
    num_classes = 3
    image_shape_wohood = (520 - 200, 800)
    streets_im = []
    for rgb_frame in video:
        start = timeit.default_timer()
        rgb_frame_t = rgb_frame[200:520, 0:800]
        rgb_frame_h = scipy.misc.imresize(rgb_frame_t, image_shape)
        rgb_frame_m = rgb_frame_h.reshape(1, image_shape[0], image_shape[1], 3)
        im_softmax = sess.run(out, {x: rgb_frame_m, keep_prob: 1.0})
        road_pix  = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])
        car_pix   = im_softmax[:, 2].reshape(image_shape[0], image_shape[1])

        road_seg  = (road_pix > 0.5).reshape(image_shape[0], image_shape[1], 1)
        car_seg  = (car_pix > 0.5).reshape(image_shape[0], image_shape[1], 1)

        #road_seg = scipy.misc.imresize(road_seg, image_shape_wohood)
        #car_seg = scipy.misc.imresize(car_seg, image_shape_wohood)

        #binary_car_result = np.pad(car_seg, ((200, 80), (0, 0)), 'constant')
        #binary_road_result = np.pad(road_seg, ((200, 80), (0, 0)), 'constant')

        #binary_road_result = binary_road_result.reshape(600, 800, 1)
        #binary_car_result = binary_car_result.reshape(600, 800, 1)

        mask = np.dot(road_seg, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(rgb_frame_h)
        street_im.paste(mask, box=None, mask=mask)
        mask = np.dot(car_seg, np.array([[255, 0, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im.paste(mask, box=None, mask=mask)
        streets_im.append(np.asarray(street_im))
        frame += 1
        stop = timeit.default_timer()
        print(frame)
        print(stop - start)
    skvideo.io.vwrite('annotated_video.mp4', streets_im)
    return output_compare

def main(_):
    data_dir = './data'
    helper.maybe_download_pretrained_vgg(data_dir)
    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    tf.import_graph_def(graph_def, name='')
    video = skvideo.io.vread('Videos/test_video.mp4')
    sess, ops = load_graph('./frozen_model_2/graph.pb')
    output_predicted = benchmark(sess, video)

if __name__ == '__main__':
    tf.app.run(main=main)
