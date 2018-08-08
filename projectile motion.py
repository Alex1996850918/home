import numpy as np
import tensorflow as tf
from math import pi
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
tStart = time.time()

g=9.8
Activation = tf.nn.tanh
Nlayer = 10
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
## NN builder
def build_NN(inn,out,norm):
    def neuron_adder(inputs,input_size,output_size,n_layer,activation_function = None,norm = False):
        W = tf.Variable(tf.random_uniform([input_size,output_size],-1,1),name='W')
        b = tf.Variable(tf.zeros([1,output_size])-0.1,name='b')
        Nout = tf.matmul(inputs,W)+b
        # Batch Normalizaion    
        if norm:
            Nout = tf.layers.batch_normalization(Nout, training=training)
        if activation_function is None:
            outputs = Nout
        else:
            outputs = activation_function(Nout)
        return outputs
    # Normalize Initial Data
    if norm:
        inn = tf.layers.batch_normalization(inn, training=training)

    layers_inputs = [inn]
    for i in range(Nlayer):
        if i <= 1:
            layer_input = layers_inputs[i]
            inputs_size = layers_inputs[i].get_shape()[1].value
            output = neuron_adder(layer_input,inputs_size,int(Nhidden),i+1,Activation,norm)
            layers_inputs.append(output)
        elif i > 1:
            layer_input = layers_inputs[i]
            inputs_size = layers_inputs[i].get_shape()[1].value
            output = neuron_adder(layer_input,inputs_size,int(Nhidden),i+1,Activation,norm)
            layers_inputs.append(output)       
    prediction = neuron_adder(layers_inputs[-1],layers_inputs[-1].get_shape()[1].value,label.shape[1],i+1,None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(out-prediction),reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): 
           train = optimizer.minimize(loss)
    return [train,loss,layers_inputs,prediction]
Xi = tf.placeholder(tf.float32,[None,feature.shape[1]])
Yi = tf.placeholder(tf.float32,[None,label.shape[1]])
## build NN
train,loss,layers_inputs,prediction = build_NN(Xi,Yi,True)
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
## train NN
sess.run(ini)
for i in range(20000):
    batch = np.random.randint(0,100)*100 #dvide my training data
    sess.run(train,feed_dict = {Xi:feature[batch:batch+100,:],Yi:label[batch:batch+100,:]})
    if i % 100 == 0:    
        cost_train.append(sess.run(loss,feed_dict = {Xi:feature[batch:batch+1000,:],Yi:label[batch:batch+1000,:]}))
        cost_test.append(sess.run(loss,feed_dict = {Xi:feature_TEST,Yi:label_TEST}))
        inter = np.hstack((feature,label))
        np.random.shuffle(inter)#rearrange my data
        feature = inter[:,0:2]#rearrnge my data
        label = inter[:,2:4]#rearrnge my data
        inter = []
    if i == 19999:
        Prediction = sess.run(prediction,feed_dict = {Xi:feature,Yi:label})
        save_path = saver.save(sess, "my_net/save_net.ckpt")
loss_TEST = sess.run(loss,feed_dict = {Xi:feature_TEST,Yi:label_TEST})
Prediction_TEST = sess.run(prediction,feed_dict = {Xi:feature_TEST,Yi:label_TEST})

plt.figure(1)
R1,p1 = pearsonr(H_norm_TEST,Prediction_TEST[:,0].reshape(H_norm_TEST.shape))
plt.scatter(H_norm_TEST,Prediction_TEST[:,0],s = 10,alpha = 0.5)
x1 = np.linspace(min(H_norm_TEST),max(H_norm_TEST),50)
y1 = x1
plt.plot(x1,y1,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$')
plt.ylabel('$computational$')
plt.title('$Maximum\ Height\ of\ Projectile$,BN & using tanh,test')
plt.annotate('R=%.4f\n10000 iteration'%(R1),xy = (1,0.),xycoords='data',fontsize = 9) 

plt.figure(2)
R2,p2 = pearsonr(R_norm_TEST,Prediction_TEST[:,1].reshape(R_norm_TEST.shape))
plt.scatter(R_norm_TEST,Prediction_TEST[:,1],s = 10,alpha = 0.5)
x2 = np.linspace(min(R_norm_TEST),max(R_norm_TEST),50)
y2 = x2
plt.plot(x2,y2,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$')
plt.ylabel('$computational$')
plt.title('$Maximum\ Distance\ of\ Projectile$,BN & using tanh,test')
plt.annotate('R=%.4f\n10000 iteration'%(R2),xy = (1,0.),xycoords='data',fontsize = 9)

plt.figure(3)
R3,p3 = pearsonr(label[:,0],Prediction[:,0].reshape(label[:,0].shape))
plt.scatter(label[:,0],Prediction[:,0],s = 10,alpha = 0.5)
x3 = np.linspace(min(label[:,0]),max(label[:,0]),50)
y3 = x3
plt.plot(x1,y1,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$')
plt.ylabel('$computational$')
plt.title('$Maximum\ Height\ of\ Projectile$,BN & using tanh,train')
plt.annotate('R=%.4f\n10000 iteration'%(R3),xy = (1,0.),xycoords='data',fontsize = 9)

plt.figure(4)
R4,p4 = pearsonr(label[:,1],Prediction[:,1].reshape(label[:,1].shape))
plt.scatter(label[:,1],Prediction[:,1],s = 10,alpha = 0.5)
x4 = np.linspace(min(label[:,1]),max(label[:,1]),50)
y4 = x4
plt.plot(x2,y2,color = 'red',linewidth = 1)
plt.xlabel('$theoratical$')
plt.ylabel('$computational$')
plt.title('$Maximum\ Distance\ of\ Projectile$,BN & using tanh,train')
plt.annotate('R=%.4f\n10000 iteration'%(R4),xy = (1,0.),xycoords='data',fontsize = 9)

plt.figure(5)
plt.plot(np.transpose(np.array([np.linspace(0,20000,200)])),np.log10(cost_train),np.transpose(np.array([np.linspace(0,20000,200)])),np.log10(cost_test))
#plt.show()
tEnd = time.time()
print(tEnd - tStart)
#f=open("BN tanh loss 5.txt","w")
#txt = np.hstack((np.transpose(np.array([np.linspace(0,10000,200)])),np.transpose(np.array([np.log10(cost_train)]))))
#np.savetxt(f,txt,'%2.9f',' ')
#f.close()

#
                   
