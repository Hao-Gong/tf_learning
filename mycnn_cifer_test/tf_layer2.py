##About the basic training of the data, y=ax+b
#!/usr/bin/sudo python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(np.sin(x_data*10))+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

l1=tf.layers.dense(xs, units=100, activation=tf.nn.relu,name="hidden_layer1")
l2=tf.layers.dense(l1, units=50, activation=tf.nn.relu,name="hidden_layer2")
l3=tf.layers.dense(l2, units=50, activation=tf.nn.relu,name="hidden_layer3")
prediction=tf.layers.dense(l3, units=1, activation=None,name="output_layer4")

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                  reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

#initialize the Variable
init=tf.global_variables_initializer()
sess=tf.Session()
##tensorboard create
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("board6/", sess.graph)
sess.run(init)

#create plt
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#make the plt show not chock
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i%10==0:
        prediction_value,result = sess.run([prediction,merged], feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
        #print(sess.run(tf.get_variable('hidden_layer1'),feed_dict={xs: x_data, ys: y_data}))
        try:
            #remove old and can create new
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #red line,width=5
        lines=ax.plot(x_data,prediction_value,'r-',lw=1)
        plt.pause(0.01)

