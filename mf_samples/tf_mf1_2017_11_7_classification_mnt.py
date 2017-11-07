##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
##import mnist examples and load it, the examples contain 55000 pic with 28*28 pixels,the input size is 784 units
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

 
###self defined layer function, the hyperparameter should be optimized in the function
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        #Matrix
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram("/Value",Weights)
        #vector
        with tf.name_scope('bias'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram("/Value",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
            
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            #the function default has default name
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram("/Value",outputs)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

###the main function

xs=tf.placeholder(tf.float32,[None,784])# input 28*28
ys=tf.placeholder(tf.float32,[None,10])#0-9 classification

#softmax is suitable for classification, and relu is suitable for the regression calculation
prediction=add_layer(xs,784,10,1,activation_function=tf.nn.softmax)

#the cross_entropy to calculate the classification
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize the Variable
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    #you need't use all the samples in the sets
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs: batch_xs,ys: batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

