import os
os.environ["CHAINER_TYPE_CHECK"] = "0"


import numpy as np
import imp

# from chainer import function
# from chainer import variable
# from chainer.functions.array.get_item import GetItem
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D
from chainer.functions.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from chainer.functions.loss.sigmoid_cross_entropy import SigmoidCrossEntropy
# from chainer.functions.normalization.batch_normalization import BatchNormalizationFunction
# from chainer.functions.array.reshape import Reshape
# from chainer.functions.activation.relu import ReLU
# from chainer.functions.noise.dropout import Dropout
# from chainer.functions.noise.gaussian import Gaussian
# from chainer.functions.activation.sigmoid import Sigmoid
from chainer.functions.connection.linear import LinearFunction
from chainer.functions.normalization.local_response_normalization import LocalResponseNormalization
from chainer.functions.array.concat import Concat
from chainer.functions.math.basic_math import Add
from rank_multi_port import get_node_list


#delete_node_set = [ReLU, Dropout, Reshape, BatchNormalizationFunction]

build_node_set = [MaxPooling2D, AveragePooling2D, Convolution2DFunction, LinearFunction, Concat, Add]
end_node_set = [SigmoidCrossEntropy, SoftmaxCrossEntropy]





class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)


def convert_to_node(model_info):
    node_info = []
    for i in range(0,len(model_info)):
        temp_node = Node(**model_info[i])
        node_info.append(temp_node)
    return node_info


def get_chainer_convolution_output(input_size, n, k, p, s, cover_all):
    # input_size: the shape of input data
    # n: the num of output
    # k: kernel_size
    # p: pad
    # s: stride
    # cover_all: default 'False'

    size = input_size[1]
    num_output = n

    if cover_all:
        out_size = (size + p*2 - k + s - 1)//s + 1
    else:
        out_size = (size + p*2 - k)//s + 1

    pad_whole_size = size + p*2

    pad_in_size = (out_size - 1)*s + k

    if pad_in_size > pad_whole_size :pad_whole_size = pad_in_size

    pad_whole = (input_size[0], pad_whole_size, pad_whole_size)
    pad_in = (input_size[0], pad_in_size, pad_in_size)

    output_size = (num_output, out_size, out_size)

    return (output_size, pad_whole, pad_in)

# def get_all_node(input_node):
#     temp_node = input_node
#     model_info.append(temp_node)
#     while len(temp_node.output_nodes) != 0:
#         if len(temp_node.output_nodes) > 1:
#             for one_node in temp_node.output_nodes:
#                 get_all_node(one_node)
#             break
#         else:
#             temp_node = temp_node.output_nodes[0]
#             model_info.append(temp_node)

def get_rank0(func_node):
    for one_node in func_node:
        if one_node.rank == 0:
            return one_node

def isdelnode(func_node):
    for i in build_node_set:
        if isinstance(func_node, i):
            return False
    return True

def isendnode(func_node):
    for i in end_node_set:
        if isinstance(func_node, i):
            return True

    return False


def build_input_output_name(model_info):
    for one_node in model_info:
        tmp_node = one_node['node']
        input_name=[]
        output_name = []
        for tmp_in in tmp_node.input_nodes:
            input_name.append(tmp_in.name)
        for tmp_out in tmp_node.output_nodes:
            output_name.append(tmp_out.name)

        one_node['input_name'] = input_name
        one_node['output_name'] = output_name

    return model_info


# def build_nodes(node_info):
#
#     for one_node in node_info:
#         input_nodes = []
#         output_nodes = []
#         for one_input in one_node.input_name:
#             tmp = search_node(node_info, one_input)
#             input_nodes.append(tmp)
#
#         for one_output in one_node.output_name:
#             tmp = search_node(node_info, one_output)
#             output_nodes.append(tmp)
#
#         one_node.input_nodes = input_nodes
#         one_node.output_nodes = output_nodes
#
#     return node_info


def get_chainer_raw_model(chainer_file, class_name):
    # model = alex.Alex()
    # model = googlenet.GoogLeNet()
    # model = resnet.ResNet()
    # model = resnet50.ResNet50()
    file_name = chainer_file.split('/')[-1]
    file_name = file_name.split('.')[0]
    foo = imp.load_source(file_name, chainer_file)
    class_ = getattr(foo, class_name)
    model = class_()
    data = np.ndarray((128,3,model.insize,model.insize), dtype=np.float32)
    data.fill(3333)

    label = np.ndarray((128), dtype=np.int32)
    label.fill(1)

    x = np.asarray(data)
    y = np.asarray(label)
    outputs = model(x, y)

    output_node = []

    def get_output_node(one_var):
        temp_var = one_var

        if temp_var.creator and (not isendnode(temp_var.creator)):
            for one_var in temp_var.creator.inputs:
                get_output_node(one_var)

        elif temp_var.creator and (isendnode(temp_var.creator)):

            output_node.append(temp_var.creator.inputs[0])

    get_output_node(outputs)
    output_set = output_node

    functionnode = []

    def add_inputnode(one_var):
        temp_output_nodes = one_var
        if temp_output_nodes.creator and temp_output_nodes.creator not in functionnode:
            temp_func_node = temp_output_nodes.creator
            temp_func_node.input_nodes = []
            temp_func_node.output_nodes = []

            for one_input in temp_func_node.inputs:
                if one_input.creator:
                    temp_func_node.input_nodes.append(one_input.creator)
            functionnode.append(temp_func_node)
            for one_var in temp_func_node.inputs:
                add_inputnode(one_var)

    for one_out in output_set:
        add_inputnode(one_out)

    # build the output_nodes
    for one_node in functionnode:
        temp_output_nodes = []
        for temp_node in functionnode:
            if one_node in temp_node.input_nodes:
                temp_output_nodes.append(temp_node)
        one_node.output_nodes = temp_output_nodes

    # get the name format of node and build the scan order.
    for one_func_node in functionnode:
        if isinstance(one_func_node, Convolution2DFunction):
            one_func_node.name = 'conv' + '_' + str(one_func_node.rank) + '_' + str(
            functionnode.index(one_func_node))
        elif isinstance(one_func_node, MaxPooling2D):
            one_func_node.name = 'pool' + '_' + str(one_func_node.rank) + '_' + str(
            functionnode.index(one_func_node))
        elif isinstance(one_func_node, AveragePooling2D):
            one_func_node.name = 'pool' + '_' + str(one_func_node.rank) + '_' + str(
            functionnode.index(one_func_node))
        elif isinstance(one_func_node, Add):
            one_func_node.name = 'add' + '_' + str(one_func_node.rank) + '_' + str(
            functionnode.index(one_func_node))
        elif isinstance(one_func_node, Concat):
            one_func_node.name = 'concat' + '_' + str(one_func_node.rank) + '_' + str(
                functionnode.index(one_func_node))
        elif isinstance(one_func_node, LinearFunction):
            one_func_node.name = 'fc' + '_' + str(one_func_node.rank) + '_' + str(
                functionnode.index(one_func_node))
        else:
            one_func_node.name = str(one_func_node.rank) + '_' + str(
                functionnode.index(one_func_node))


    new_func_nodes = []
    for one_func_node in functionnode:
        # updated the node set that should be deleted
        if isdelnode(one_func_node):
            input_node = one_func_node.input_nodes[0]
            output_nodes = one_func_node.output_nodes
            input_node.output_nodes = output_nodes

            for i in range(0,len(output_nodes)):
                try:
                    output_nodes[i].input_nodes.remove(one_func_node)
                except:
                    print "Error"
                output_nodes[i].input_nodes.append(input_node)

            continue
        new_func_nodes.append(one_func_node)


    node_info = []
    for one_node in new_func_nodes:
        dict = {}
        dict['name'] = one_node.name
        dict['label'] = one_node.label
        dict['rank'] = one_node.rank
        dict['node'] = one_node
        dict['input_size'] = one_node.inputs[0].shape[1:]
        if isinstance(one_node, Convolution2DFunction):
            dict['group'] = 1
            dict['type'] = 'Convolution'
            dict['pad'] = one_node.ph
            dict['stride'] = one_node.sx
            dict['ph'] = one_node.ph
            dict['pw'] = one_node.pw
            dict['sh'] = one_node.sy
            dict['sw'] = one_node.sx
            dict['kh'] = one_node.inputs[1].shape[2]
            dict['kw'] = one_node.inputs[1].shape[3]
            # get the kernel_size from the weight variable
            dict['kernel_size'] = one_node.inputs[1].shape[3]
            dict['num_output'] = one_node.inputs[1].shape[0]
            cover_all = one_node.cover_all
            dict['cover_all'] = cover_all
            dict['output_size'], dict['pad_whole'], dict['pad_in'] = \
                get_chainer_convolution_output(dict['input_size'],
                                               dict['num_output'],
                                               dict['kernel_size'],
                                               dict['pad'],
                                               dict['stride'],
                                               cover_all)
        elif isinstance(one_node, MaxPooling2D) or isinstance(one_node, AveragePooling2D):
            if isinstance(one_node, MaxPooling2D):
                dict['type'] = 'MaxPooling'
            if isinstance(one_node, AveragePooling2D):
                dict['type'] = 'AveragePooling'
            dict['pad'] = one_node.ph
            dict['stride'] = one_node.sx
            dict['kernel_size'] = one_node.kh

            dict['ph'] = one_node.ph
            dict['pw'] = one_node.pw
            dict['sh'] = one_node.sy
            dict['sw'] = one_node.sx
            dict['kh'] = one_node.kh
            dict['kw'] = one_node.kw

            dict['num_output'] = one_node.inputs[0].shape[1]
            cover_all = one_node.cover_all
            dict['cover_all'] = cover_all
            dict['output_size'], dict['pad_whole'], dict['pad_in'] = \
                get_chainer_convolution_output(dict['input_size'],
                                               dict['num_output'],
                                               dict['kernel_size'],
                                               dict['pad'],
                                               dict['stride'],
                                               cover_all)
        elif isinstance(one_node, Add):
            dict['type'] = 'Eltwise'
            dict['output_size'] = one_node.inputs[0].shape[1:]

        elif isinstance(one_node, Concat):
            dict['type'] = 'Concat'
            dict['output_size'] = (0, one_node.inputs[0].shape[2], one_node.inputs[0].shape[3])
            for one_input_var in one_node.inputs:
                temp_output_size = list(dict['output_size'])
                temp_output_size[0] += one_input_var.shape[1]
                dict['output_size'] = tuple(temp_output_size)
        elif isinstance(one_node, LinearFunction):
            temp_input_size = 1
            for i in dict['input_size']:
                temp_input_size *= i
            dict['input_size'] = (temp_input_size,)
            dict['type'] = 'InnerProduct'
            dict['output_size'] = (one_node.inputs[2].shape[0],)

        elif isinstance(one_node, LocalResponseNormalization):
            dict['type'] ='LRN'
            dict['output_size'] = dict['input_size']

        node_info.append(dict)

    return node_info


def get_chainer_model(chainer_file, class_name):


    model_info = get_chainer_raw_model(chainer_file, class_name)
    model_info = build_input_output_name(model_info)

    return model_info[::-1]

# # chainer_file = '/Users/Mac/zhanGGe/dl_model/train_googlenet.py'
# chainer_file = '/Users/Mac/zhanGGe/dl_model/resnet50.py'
# chainer_file = '/home/zhangge/pymodel/alex_new.py'
# # class_name = 'GoogLeNet'
# class_name = 'Alex'
#
# model_info = get_chainer_model(chainer_file, class_name)
# node_info = get_node_list(model_info)


