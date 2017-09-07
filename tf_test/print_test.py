#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('test_model1.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))


# print(tf.get_default_graph().get_all_collection_keys())
# op = sess.graph.get_operations()
# all_vars = tf.Graph.get_operations()
# print(op)
for i in tf.get_default_graph().get_operations():
    print i.type
    if i.type == 'Relu' or i.type == "MaxPool":
    # print i.type
    # print ('-----------')
    # print i.outputs
    # print ('-----------')
    # print i.inputs[0]
    # print ('-----------')
    # print i.node_def
        print ("===============================")
        for j in i.inputs:
            print ("inputs", j)
        print ("name", i.name)
        print ("node_def\n", i.node_def)
        for j in i.outputs:
            print ("outputs", j)
        print ("type", i.type)



# g = tf.get_default_graph()
#
#
# print g.get_all_collection_keys()
#
# for i in g.get_collection('trainable_variables'):
#     print (i.op, i.name )
#     print i


