#this is learned from MOFAN, LINK: https://morvanzhou.github.io/
##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        #Matrix
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+"/weights",Weights)
        #vector
        with tf.name_scope('bias'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+"/biases",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
            
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            #the function default has default name
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/weights",outputs)
        return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(1-x_data*x_data)+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
l2=add_layer(l1,10,20,n_layer=2,activation_function=tf.nn.relu)
l3=add_layer(l2,20,10,n_layer=3,activation_function=tf.nn.relu)
prediction=add_layer(l3,10,1,n_layer=4,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                  reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

#initialize the Variable
sess=tf.Session()

merged=tf.summary.merge_all()
#writer=tf.summary.FileWriter("logs/", sess.graph)
writer=tf.summary.FileWriter("board/", sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
