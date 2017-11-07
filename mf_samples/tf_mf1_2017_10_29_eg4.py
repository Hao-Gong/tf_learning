##About the basic training of the data, y=ax+b

import tensorflow as tf
import numpy as np

#create data
state=tf.Variable(0,name='counter')
#print(state.name)

one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.global_variables_initializer()#if defined variable
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
