#this is learned from MOFAN, LINK: https://morvanzhou.github.io/
##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        #Matrix
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        #vector
        with tf.name_scope('bias'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
            
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            #the function default has default name
            outputs=activation_function(Wx_plus_b)
        return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(1-x_data*x_data)+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
l2=add_layer(l1,10,20,activation_function=tf.nn.relu)
l3=add_layer(l2,20,10,activation_function=tf.nn.relu)
prediction=add_layer(l3,10,1,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                  reduction_indices=[1]))
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

#initialize the Variable
sess=tf.Session()
sess.run(tf.global_variables_initializer())


#tf.summary.merge_all()
#writer=tf.summary.FileWriter("logs/", sess.graph)
writer=tf.summary.FileWriter("board/", sess.graph)



