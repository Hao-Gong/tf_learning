##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np

#create data
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(outpu,feed_dict={input1:[7.],input2:[2.]}))
