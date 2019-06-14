import tensorflow as tf
import os
import numpy as np
import ten_vgg16 as VGG16
import cv2 as cv

IMAGENET_MEAN=[104.051,112.514,116.676]

batch_size=100
learning_rate=0.001
regularization_rate=0.00001
training_step=10000
num_class=2

def mean_images(im_f):
    img=cv.imread(im_f)
    img_resize=cv.resize(img,())
    img_resize=img_resize.astype(np.float32)
    img_resize-=IMAGENET_MEAN
    processed_img=np.reshape(img_resize,[])
    return processed_img




def train():
    x=tf.placeholder(tf.float32,[batch_size,imge_size,image_size,3],name='x_input')
    y_=tf.placeholder(tf.float32,[None,num_class],name='y_output')

    y=VGG16(X=x,NUM_CLASSES=num_class)
    global_step=tf.Variable(0,trainable=False)

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

    correct_pred = tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))











