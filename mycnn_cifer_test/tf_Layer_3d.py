##About the basic training of the data, y=ax+b
#!/usr/bin/sudo python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(np.sin(x_data*10))+noise


xs=tf.constant(x_data,dtype=tf.float32)
ys=tf.constant(y_data,dtype=tf.float32)

l1=tf.layers.dense(xs, units=100, activation=tf.nn.relu,name="hidden_layer1")
l2=tf.layers.dense(l1, units=50, activation=tf.nn.relu,name="hidden_layer2")
l3=tf.layers.dense(l2, units=50, activation=tf.nn.relu,name="hidden_layer3")
prediction=tf.layers.dense(l3, units=1, activation=None,name="output_layer4")

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                  reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

#initialize the Variable
init=tf.global_variables_initializer()
sess=tf.Session()
##tensorboard create
merged=tf.summary.merge_all()
#writer=tf.summary.FileWriter("board4/", sess.graph)
sess.run(init)

#create plt
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#make the plt show not chock
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step)
    if i%10==0:
        print(sess.run(loss))
        prediction_value=sess.run(prediction)

        #writer.add_summary(summary, i)
        try:
            #remove old and can create new
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #red line,width=5
        lines=ax.plot(x_data,prediction_value,'r-',lw=1)
        plt.pause(0.01)


