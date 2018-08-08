# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:15:46 2018

@author: Alex Lee
"""
import numpy as np
import tensorflow as tf
from math import pi
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

g=9.8
Activation = tf.nn.sigmoid
Nlayer = 5
Nhidden = 10
training = True
## initailize angle
th = np.random.rand(10000,1).astype(np.float32)*pi/2
th_sin = np.sin(th)
th_sin2 = np.sin(2*th)

th_TEST = np.random.rand(2000,1).astype(np.float32)*pi/2
th_sin_TEST = np.sin(th_TEST)
th_sin2_TEST = np.sin(2*th_TEST)
## initailize velocity
v = np.random.rand(10000,1).astype(np.float32)*10
v_TEST = np.random.rand(2000,1).astype(np.float32)*10
## height & horizontal displacement
H = v*v*th_sin*th_sin/2/g
R = v*v*th_sin2/g
H_mean = np.mean(H)
H_std = np.std(H)
H_norm=(H-H_mean)/H_std
R_mean = np.mean(R)
R_std = np.std(R)
R_norm=(R-R_mean)/R_std

H_TEST = v_TEST*v_TEST*th_sin_TEST*th_sin_TEST/2/g
R_TEST = v_TEST*v_TEST*th_sin2_TEST/g
H_mean_TEST = np.mean(H_TEST)
H_std_TEST = np.std(H_TEST)
H_norm_TEST = (H_TEST-H_mean_TEST)/H_std_TEST
R_mean_TEST = np.mean(R_TEST)
R_std_TEST = np.std(R_TEST)
R_norm_TEST =(R_TEST-R_mean_TEST)/R_std_TEST
## data prepare
feature = np.hstack((th,v))
label = np.hstack((H_norm,R_norm))
feature_TEST = np.hstack((th_TEST,v_TEST))
label_TEST = np.hstack((H_norm_TEST,R_norm_TEST))
def build_NN(inputs,norm):
    def neuron_adder(inputs,Nhidden,activation_function = None,norm = False):
        if norm:
            inputs = tf.layers.batch_normalization(inputs,training=training)
        outputs = tf.layers.dense(inputs,Nhidden)
        if norm:
            outputs = tf.layers.batch_normalization(outputs,training=training)
        if activation_function is None:
            return outputs
        else:
            outputs = activation_function(outputs)
            return outputs
    for i in range(Nlayer):
        N = neuron_adder(inputs,Nhidden,Activation,norm)
        inputs = N
    outputs = tf.layers.dense(inputs,label.shape[1])
    return outputs
##
train_data = tf.data.Dataset.from_tensor_slices((feature,label)).repeat().batch(100).shuffle(300)
train_iterator = train_data.make_one_shot_iterator()
test_data = tf.data.Dataset.from_tensor_slices((feature_TEST,label_TEST)).repeat().batch(2000)
test_iterator = test_data.make_one_shot_iterator()
train_data_1 = tf.data.Dataset.from_tensor_slices((feature,label)).repeat().batch(10000)
train_1_iterator = train_data_1.make_one_shot_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
next_element = iterator.get_next()

prediction = build_NN(next_element[0],True)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(next_element[1]-prediction),reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer()
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): 
     train = optimizer.minimize(loss)
ini = tf.global_variables_initializer()
sess = tf.Session() 
cost_train = []
cost_test = []
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5) 
##
train_iter_handle = sess.run(train_iterator.string_handle())
test_iter_handle = sess.run(test_iterator.string_handle())
train_iter_handle_1 = sess.run(train_1_iterator.string_handle())
sess.run(ini)
for j in range(20000):
    a,b = sess.run((train,loss),feed_dict = {handle:train_iter_handle})
    if j % 100 == 0:
        cost_train.append(b)
        c = sess.run(loss,feed_dict = {handle:test_iter_handle})
        cost_test.append(c)
    if j == 19999:
        Predict = sess.run(prediction,feed_dict = {handle:train_iter_handle_1})
        save_path = saver.save(sess, "my_net/save_net.ckpt")
Predict_TEST = sess.run(prediction,feed_dict = {handle:test_iter_handle})

plt.figure(1)
R1,p1 = pearsonr(H_norm_TEST,Predict_TEST[:,0].reshape(H_norm_TEST.shape))
plt.scatter(H_norm_TEST,Predict_TEST[:,0],s = 10,alpha = 0.5)
x1 = np.linspace(min(H_norm_TEST),max(H_norm_TEST),50)
y1 = x1
plt.plot(x1,y1,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$', fontsize=15)
plt.ylabel('$computational$', fontsize=15)
plt.title('$Maximum\ Height\ of\ Projectile$,BN & using sigmoid,test', fontsize=20)
plt.annotate('R=%.4f\n20000 iteration'%(R1),xy = (1,0.),xycoords='data',fontsize = 15) 

plt.figure(2)
R2,p2 = pearsonr(R_norm_TEST,Predict_TEST[:,1].reshape(R_norm_TEST.shape))
plt.scatter(R_norm_TEST,Predict_TEST[:,1],s = 10,alpha = 0.5)
x2 = np.linspace(min(R_norm_TEST),max(R_norm_TEST),50)
y2 = x2
plt.plot(x2,y2,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$', fontsize=15)
plt.ylabel('$computational$', fontsize=15)
plt.title('$Maximum\ Distance\ of\ Projectile$,BN & using sigmoid,test', fontsize=20)
plt.annotate('R=%.4f\n20000 iteration'%(R2),xy = (1,0.),xycoords='data',fontsize = 15)

plt.figure(3)
R3,p3 = pearsonr(label[:,0],Predict[:,0].reshape(label[:,0].shape))
plt.scatter(label[:,0],Predict[:,0],s = 10,alpha = 0.5)
x3 = np.linspace(min(label[:,0]),max(label[:,0]),50)
y3 = x3
plt.plot(x1,y1,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$', fontsize=15)
plt.ylabel('$computational$', fontsize=15)
plt.title('$Maximum\ Height\ of\ Projectile$,BN & using sigmoid,train', fontsize=20)
plt.annotate('R=%.4f\n20000 iteration'%(R3),xy = (1,0.),xycoords='data',fontsize = 15)

plt.figure(4)
R4,p4 = pearsonr(label[:,1],Predict[:,1].reshape(label[:,1].shape))
plt.scatter(label[:,1],Predict[:,1],s = 10,alpha = 0.5)
x4 = np.linspace(min(label[:,1]),max(label[:,1]),50)
y4 = x4
plt.plot(x2,y2,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$', fontsize=15)
plt.ylabel('$computational$', fontsize=15)
plt.title('$Maximum\ Distance\ of\ Projectile$,BN & using sigmoid,train', fontsize=20)
plt.annotate('R=%.4f\n20000 iteration'%(R4),xy = (1,0.),xycoords='data',fontsize = 15)

plt.figure(5)
plt.plot(np.transpose(np.array([np.linspace(0,20000,200)])),np.log10(cost_train),'b',np.transpose(np.array([np.linspace(0,20000,200)])),np.log10(cost_test),'r')
plt.xlabel('$training step$', fontsize=20)
plt.ylabel('$loss(log scale)$', fontsize=20)
plt.title('$Loss,BN\ &\  using\ sigmoid$', fontsize=20)
plt.legend(['train','test'],fontsize = 15)

plt.show()

           
       
    
