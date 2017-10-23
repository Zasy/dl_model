import chainer.links as L
import alex
import numpy as np

from chainer import function
from chainer import variable
from chainer.functions.array.get_item import GetItem
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D
from chainer.functions.normalization.batch_normalization import BatchNormalization
from chainer.functions.array.reshape import Reshape
from chainer.functions.activation.relu import ReLU
from chainer.functions.noise.dropout import Dropout
from chainer.functions.noise.gaussian import Gaussian
from chainer.functions.activation.sigmoid import Sigmoid
from chainer.functions.connection.linear import LinearFunction
from chainer.functions.normalization.local_response_normalization import LocalResponseNormalization

from chainer.functions.array.concat import Concat
from chainer.functions.math.basic_math import Add

import googlenet
import resnet
import resnet50
delete_node_set = [ReLU, Dropout, Reshape, BatchNormalization]

build_node_set = [MaxPooling2D, AveragePooling2D, Convolution2DFunction, LinearFunction,
                  LocalResponseNormalization, Concat, Add]


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
    pad_whole = (input_size[0], pad_whole_size, pad_whole_size)
    pad_in_size = (out_size - 1)*s
    pad_in = (input_size[0], pad_in_size, pad_in_size)
    output_size = (num_output, out_size, out_size)

    return (output_size, pad_whole, pad_in)

def get_all_node(input_node):
    temp_node = input_node
    model_info.append(temp_node)
    while len(temp_node.output_nodes) != 0:
        if len(temp_node.output_nodes) > 1:
            for one_node in temp_node.output_nodes:
                get_all_node(one_node)
            break
        else:
            temp_node = temp_node.output_nodes[0]
            model_info.append(temp_node)

def get_rank0(func_node):
    for one_node in func_node:
        if one_node.rank == 0:
            return one_node

def isdelnode(func_node):
    for i in delete_node_set:
        if isinstance(func_node, i):
            return True
    return False

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

def get_chainer_model(model):
    # model = alex.Alex()
    # model = googlenet.GoogLeNet()
    # model = resnet.ResNet()
    model = resnet50.ResNet50()

    data = np.ndarray((128,3,model.insize,model.insize), dtype=np.float32)
    data.fill(3333)

    label = np.ndarray((128), dtype=np.int32)
    label.fill(1)

    x = np.asarray(data)
    functionnode = []
    outputs = model.forward(x)
    output_set = []

    # delete the getitem problom
    # and get the output_set
    for o in outputs:
        if isinstance(o, variable.Variable):
            o = o.node
        while isinstance(o.creator, GetItem):
            o = o.creator.inputs[0]
        if o not in output_set:
            output_set.append(o)
    # get every node from the out of forward
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
        one_func_node.name = one_func_node.label + '/' + str(one_func_node.rank) + '/' + str(
            functionnode.index(one_func_node))
        if len(one_func_node.input_nodes) > 1:
            for i in range(0,len(one_func_node.input_nodes) - 1):
                one_func_node.input_nodes[i].output_nodes.remove(one_func_node)
            one_func_node.input_nodes = [one_func_node.input_nodes[-1]]

    new_func_nodes = []
    for one_func_node in functionnode:
        # updated the node set that should be deleted
        if isdelnode(one_func_node):
            input_node = one_func_node.input_nodes[0]
            output_nodes = one_func_node.output_nodes
            input_node.output_nodes = output_nodes
            for one_out in output_nodes:
                one_out.input_nodes = one_func_node.input_nodes
            continue
        new_func_nodes.append(one_func_node)


    first_node = get_rank0(new_func_nodes)
    model_info = []
    get_all_node(first_node)
    node_info = []
    for one_node in model_info:
        dict = {}
        dict['name'] = one_node.name
        dict['label'] = one_node.label
        dict['rank'] = one_node.rank
        dict['node'] = one_node
        dict['input_size'] = one_node.inputs[0].shape[1:]
        if isinstance(one_node, Convolution2DFunction):
            dict['type'] = 'Convolution'
            dict['pad'] = one_node.ph
            dict['stride'] = one_node.sx
            # get the kernel_size from the weight variable
            dict['kernel_size'] = one_node.inputs[1].shape[3]
            dict['num_of_output'] = one_node.inputs[1].shape[0]
            cover_all = one_node.cover_all
            dict['cover_all'] = cover_all
            dict['output_size'], dict['pad_in'], dict['pad_whole'] = \
                get_chainer_convolution_output(dict['input_size'],
                                               dict['num_of_output'],
                                               dict['kernel_size'],
                                               dict['pad'],
                                               dict['stride'],
                                               cover_all)
        elif isinstance(one_node, MaxPooling2D) or isinstance(one_node, AveragePooling2D):
            dict['type'] = 'Pooling'
            dict['pad'] = one_node.ph
            dict['stride'] = one_node.sx
            dict['kernel_size'] = one_node.kh
            dict['num_of_output'] = one_node.inputs[0].shape[1]
            cover_all = one_node.cover_all
            dict['cover_all'] = cover_all
            dict['output_size'], dict['pad_in'], dict['pad_whole'] = \
                get_chainer_convolution_output(dict['input_size'],
                                               dict['num_of_output'],
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
                dict['output_size'][0] += one_input_var.shape[1]

        elif isinstance(one_node, LinearFunction):
            dict['type'] = 'InnerProduct'
            dict['output_size'] = one_node.inputs[1].shape

        elif isinstance(one_node, LocalResponseNormalization):
            dict['type'] ='LRN'
            dict['output_size'] = dict['input_size']

        node_info.append(dict)

    return node_info


