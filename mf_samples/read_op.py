import tensorflow as tf
import numpy as np
#you still need to recontruct the net work

W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

#donnot need initial step

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"save_net/save_net.ckpt")
    print("weights",sess.run(W))
    print("biases",sess.run(b))