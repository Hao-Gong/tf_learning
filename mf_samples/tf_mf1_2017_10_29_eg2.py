##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

###create tensoflow structure start###
#using GPU
#with tf.device('/gpu:0'):
##generate W matrix randomly

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases
loss=tf.reduce_mean(tf.square(y-y_data))

#the Optimizer machine, the learning rate is 0.5
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
###create tensoflow structure end###


#initialize the Variable
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20==0:
        print(step,sess.run(Weights),sess.run(biases))
