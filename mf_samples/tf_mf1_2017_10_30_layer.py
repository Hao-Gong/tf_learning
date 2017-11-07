##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):
    #Matrix
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #vector
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        putputs=activation_function(Wx_plus_b)
    return outputs
