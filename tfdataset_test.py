# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:03:22 2018

@author: Alex Lee
"""
import numpy as np
import tensorflow as tf

a = np.random.rand(10,2)
b = a

A = tf.placeholder(tf.int32)
Xi = tf.identity(A)


ini = tf.global_variables_initializer()
sess = tf.Session()
label_data = tf.data.Dataset.from_tensor_slices(a).repeat()
label_data = label_data.batch(sess.run(Xi,feed_dict = {A:1})).shuffle(10)
B_data = tf.data.Dataset.from_tensor_slices(b).repeat().batch(5)
label_iterator = label_data.make_initializable_iterator()
B_iterator = B_data.make_one_shot_iterator()


#handle = tf.placeholder(tf.string, shape=[])
#iterator = tf.data.Iterator.from_string_handle(handle, label_data.output_types)
next_element = label_iterator.get_next()
#label_iterator_handle = sess.run(label_iterator.string_handle())
#B_iterator_handle = sess.run(B_iterator.string_handle())
sess.run(ini)
sess.run(label_iterator.initializer)
for i in range(5):
    if i <=2:
        v = sess.run((next_element))
        print(v)
    else:
        sess.run(label_iterator.initializer)
        label_data = label_data.batch(sess.run(Xi,feed_dict = {A:5}))
        v = sess.run((next_element))
        print(v)
        
            
