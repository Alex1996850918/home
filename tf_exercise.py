import numpy as np
import tensorflow as tf


x1=np.random.rand(100).astype(np.float32)
x2=np.random.rand(100).astype(np.float32)
y=0.8*x1**2+0.2*x2+3

W=tf.Variable(tf.random_uniform([2,1],-1,1))
b=tf.Variable(tf.zeros([1]))
Y=W[0]*x1+W[1]*x2+b

loss=tf.reduce_mean(tf.square(Y-y))
optimizer=tf.train.AdamOptimizer(0.3)

ini=tf.initialize_all_variables()
train=optimizer.minimize(loss)
sess=tf.Session()
sess.run(ini)

print("step   ","w1   ","w2   ","b   ","loss")
for i in range(201):
    sess.run(train)
    if i%10==0:
        print(i,sess.run(W[0]),sess.run(W[1]),sess.run(b),sess.run(loss))
