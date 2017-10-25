import os
import importlib
import imp

from load_chainer_model import  get_chainer_model
from test_caffe import get_caffe_model
class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)


class exitMessage(object):

    def __init__(self, index, node_a, node_b, type):
        self.index = index
        self.a = node_a
        self.b = node_b
        self.type = type
        #self.param = param

class sizeMessage(object):

    def __init__(self, index, node_a, node_b, type):
        self.index = index
        self.a = node_a
        self.b = node_b
        self.type = type
        #self.param = param

def convert_to_node(model_info):
    node_info = []
    for i in range(0,len(model_info)):
        temp_node = Node(**model_info[i])
        node_info.append(temp_node)
    return node_info

def compare_type(node_a, node_b):
    return (node_a.type == node_b.type)

def compare_param(node_a, node_b):
    if node_a.type == 'Convolution' or node_a.type == 'Pooling':
        return (node_a.pad == node_b.pad and
                node_a.kernel_size == node_b.kernel_size and
                node_a.stride == node_b.stride and
                node_a.output_size == node_b.output_size and
                node_a.pad_in == node_b.pad_in and
                node_a.pad_whole == node_b.pad_whole)

    elif node_a.type == 'Concat' or node_a.type == 'Eltwise' or node_a.type == 'InnerProduct':

        return (node_a.output_size == node_b.output_size)
    else:
        return (node_a.input_size == node_b.input_size and
                node_a.output_size == node_b.output_size)

Convolution_or_pooling_str = '(%s)%s: (shape=%s, k=%d, s=%d, p=%d, p_in=%s, p_whole=%s)\n'
other_str = '(%s)%s: (shape=%s)\n'

def get_different_str(node, type):

    if type == 'Convolution' or type == 'Pooling':
        return Convolution_or_pooling_str % (node.name, node.type, node.output_size,
                                             node.kernel_size, node.stride, node.pad,
                                             node.pad_in, node.pad_whole)
    else:
        return other_str % (node.name, node.type, node.output_size)



# net_file = '/home/zhangge/caffe/models/default_resnet_50/train_val.prototxt'
net_file = '/home/zhangge/caffe/models/bvlc_alexnet/train_val.prototxt'


caffe_model_info = get_caffe_model(net_file, (3,224,224))
caffe_model_info = caffe_model_info[1:]
caffe_node_info = convert_to_node(caffe_model_info)

chainer_file = 'alex.py'
chainer_model_info = get_chainer_model(chainer_file)
chainer_node_info = convert_to_node(chainer_model_info)

exit_msg = []
size_msg = []
for i in range(0, len(chainer_node_info)):
    if not compare_type(caffe_node_info[i], chainer_node_info[i]):
        temp = exitMessage(i, caffe_node_info[i], chainer_node_info[i], caffe_node_info[i].type)
        exit_msg.append(temp)
        break

    if not compare_param(caffe_node_info[i], chainer_node_info[i]):
        temp = sizeMessage(i, caffe_node_info[i], chainer_node_info[i], caffe_node_info[i].type)
        size_msg.append(temp)

size_file_a = open('size_message_a.txt','w')
size_file_b = open('size_message_b.txt', 'w')
log_out_str = ''
exit_log_str = ''
size_log_str_a = ''
size_log_str_b = ''

if len(exit_msg) == 0:
    log_out_str += 'Topo Test Result: \n' \
                   '\tThe topolopy is SAME.'
else:
    log_out_str += 'Topo Test Result: \n' \
                   '\tThe topolopy is DIFFERENT. \n'
    log_out_str += 'The diffrence start from :\n' \
                   'node_a: ' + get_different_str(exit_msg[0].a, exit_msg[0].type) +\
                   'node_b: ' + get_different_str(exit_msg[0].b, exit_msg[0].type)

if len(size_msg) == 0:
    log_out_str += 'Size Test Result: \n' \
                   '\tThe Size of model is SAME.'
else:
    log_out_str += 'Size Test Result: \n' \
                   '\tThe Size of model is DIFFERENT. \n' \
                   '\tThe detail on the size Message file. \n'
for s_msg in size_msg:
    size_log_str_a += get_different_str(s_msg.a, s_msg.type)
    size_log_str_b += get_different_str(s_msg.b, s_msg.type)

size_file_a.write(size_log_str_a)
size_file_b.write(size_log_str_b)

size_file_a.close()
size_file_b.close()

print log_out_str







