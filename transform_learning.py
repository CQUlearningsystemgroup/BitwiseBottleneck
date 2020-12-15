# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
import os


FILE_PATH_old = ".../BIB_ImageNet/model/old/model.ckpt-2509427"
FILE_PATH_new = ".../BIB_ImageNet/model/model.ckpt-2349599"
OUTPUT_FILE = ".../BIB_ImageNet/model/new/"


old_data = []
old_name = []

new_data = []
new_name = []

print('-------------------------------------------------')
print("Old Model:\n")
for var_name_old, _ in tf.contrib.framework.list_variables(FILE_PATH_old):
    var_old = tf.contrib.framework.load_variable(FILE_PATH_old, var_name_old)
    old_data.append(var_old)
    old_name.append(var_name_old)
    print(var_name_old)
print('-------------------------------------------------')
print('\n')
print('\n')

print('-------------------------------------------------')
print("New Model:\n")
for var_name_new, _ in tf.contrib.framework.list_variables(FILE_PATH_new):
    var_new = tf.contrib.framework.load_variable(FILE_PATH_new, var_name_new)
    new_data.append(var_new)
    new_name.append(var_name_new)
    print(var_name_new)
print('-------------------------------------------------')
print('\n')
print('\n')


transform_variable_list = []

print('-------------------------------------------------')
print("The Transformed Variable:\n")
for i in range(0, len(new_name)):
    for j in range(0, len(old_name)):
        if new_name[i] == old_name[j]:
            new_data[i] = old_data[j]

    print(new_name[i])
    rename = new_name[i]
    redata = new_data[i]
    if rename.find('global_step') != -1:
        renamed_var = tf.Variable(redata, name=rename, dtype=tf.int64)
    else:
        renamed_var = tf.Variable(redata, name=rename, dtype=tf.float32)

    transform_variable_list.append(renamed_var)
print('-------------------------------------------------')
print('\n')
print('\n')

def save(saver, sess, logdir):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, write_meta_graph=False)
    print('The weights have been converted to {}.'.format(checkpoint_path))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(var_list=transform_variable_list, write_version=1)
    save(saver, sess, OUTPUT_FILE)
print("done !")







