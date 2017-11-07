import tensorflow as tf

W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name="biases")

# initialize the Variable
init = tf.global_variables_initializer()
sess = tf.Session()

saver= tf.train.Saver()
sess.run(init)

save_path=saver.save(sess,"save_net/save_net.ckpt")
print("Save path",save_path)