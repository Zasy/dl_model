'''
dl_model:2017/08/17
autuor: ZheZhan
'''

import numpy as np
import sys,os
import caffe.proto.caffe_pb2 as caffe_pb2
import math
from google.protobuf import text_format
# from tabulate import tabulate
from rank_multi_port import get_node_list


del_type_set = ['LRN']



def search_dict_node(model_info, name):
    for i in range(0, len(model_info)):
        if model_info[i]['name'] == name:
            return model_info[i]
def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = value
    return pr

def search_layer(net_layers, dst_name):
    i = 0
    for i in range(len(net_layers)):
        if net_layers[i]['name'] == dst_name:
            break
    return net_layers[i]
# input_size[channel, height, width]
# return the pad_in[channel, height, width] is the size
def get_pooing_output_size(input_size, kernel_size, stride, pad=0):

    stride_h = stride
    stride_w = stride

    kernel_h = kernel_size
    kernel_w = kernel_size

    pad_h = pad
    pad_w = pad

    input_h = input_size[1]
    input_w = input_size[2]

    # print((float)(input_h + 2*pad_h - kernel_h)/stride)
    output_h = int(math.ceil((float)(input_h + 2*pad_h - kernel_h)/stride_h) + 1)
    output_w = int(math.ceil((float)(input_w + 2*pad_w - kernel_w)/stride_w) + 1)
    if pad_h or pad_w:
        if (output_h - 1)*stride >= (input_h + pad_h):
            output_h-=1
        if (output_w - 1)*stride >= (input_w + pad_w):
            output_w -= 1
    pad_whole_h = input_h + 2*pad_h
    pad_whole_w = input_w + 2*pad_w

    pad_in_h = (output_h-1)*stride_h + kernel_h
    pad_in_w = (output_w-1)*stride_w + kernel_w

    if pad_in_h > pad_whole_h :pad_whole_h = pad_in_h
    if pad_in_w> pad_whole_w: pad_whole_w = pad_in_h

    pad_whole = (input_size[0], pad_whole_h, pad_whole_w)
    pad_in = (input_size[0], (output_h-1)*stride + kernel_h, (output_w-1)*stride +kernel_w)


    output_size = (input_size[0], output_h, output_w)

    return output_size, pad_in, pad_whole

# get the convolution detail
# return:
#   output_size: the output_size of convolution_size
#   pad_in: the data_size in the convolution op
#   pad_whole : the data_size after pad
def get_convolution_output_size(input_size, num_output, kernel_size, stride, pad):
    #   inital the kernel_size, stride and pad for h and w
    stride_h = stride
    stride_w = stride

    kernel_h = kernel_size
    kernel_w = kernel_size

    pad_h = pad
    pad_w = pad

    input_h = input_size[1]
    input_w = input_size[2]

    output_h = int((float)(input_h + 2*pad_h - kernel_h)/stride_h) + 1
    output_w = int((float)(input_w + 2*pad_w - kernel_w)/stride_w) + 1

    pad_whole_h = input_h + 2*pad_h
    pad_whole_w = input_w + 2*pad_w

    pad_whole = (input_size[0], int(pad_whole_h), int(pad_whole_w))
    pad_in = (input_size[0], int((output_h-1)*stride + kernel_h), int((output_w-1)*stride +kernel_w))
    output_size = (num_output, output_h, output_w)

    return output_size, pad_in, pad_whole

def build_graph(net, input_size, phase):
    model_info = []
    Data_layer_name = None
    for layer in net.layer:
        dict = {}
        dict['type'] = layer.type
        dict['name'] = layer.name
        dict['input_name'] = []
        dict['output_name'] = []
        dict['caffe_node'] = layer
        dict['rank'] = 1
        if layer.type == 'Data':
            if layer.include[0].phase == phase:
                # dict['locate'] = True
                # dict['input_size'] = input_size
                # dict['output_size'] = input_size
                # model_info.append(dict)
                Data_layer_name = layer.name

        if layer.type == 'DummyData':
            if layer.include[0].phase == phase:
                # dict['locate'] = True
                # dict['input_size'] = input_size
                # dict['output_size'] = input_size
                # model_info.append(dict)
                Data_layer_name = layer.name

        elif layer.type == 'Input':
            dict['locate'] = True
            dict['output_size'] = input_size
            model_info.append(dict)

        elif layer.type == 'Scale':
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['Scale'] = True

        elif layer.type == 'BatchNorm':
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['BatchNorm'] = True

        elif layer.type == 'Pooling':
            if layer.pooling_param.pool == 0:
                dict['type'] = 'MaxPooling'
            else:
                dict['type'] = 'AveragePooling'
            bottom = layer.bottom[0]
            dict['input_name'].append(bottom)
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['output_name'].append(layer.name)
            dict['input_size'] = bottom_layer['output_size']

            kernel_size = 0
            stride = 1
            pad = 0
            if layer.pooling_param.kernel_size:
                kernel_size = layer.pooling_param.kernel_size
            if layer.pooling_param.pad:
                pad = layer.pooling_param.pad
            if layer.pooling_param.stride:
                stride = layer.pooling_param.stride

            dict['kernel_size'] = kernel_size
            dict['stride'] = stride
            dict['pad'] = pad

            dict['output_size'],dict['pad_in'],dict['pad_whole'] \
                = get_pooing_output_size(dict['input_size'], kernel_size, stride, pad)
            model_info.append(dict)


        elif layer.type == 'LRN':

            bottom = layer.bottom[0]
            dict['input_name'].append(bottom)
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['output_name'].append(layer.name)

            bottom_layer['output_name'].append(layer.name)
            # print bottom_layer['output_name']
            dict['input_size'] = bottom_layer['output_size']
            dict['output_size'] = dict['input_size']
            model_info.append(dict)


        elif layer.type == 'Convolution':

            bottom = layer.bottom[0]
            if bottom == Data_layer_name:
                pass
                dict['rank'] = 0
                dict['input_size'] = input_size
            else:
                bottom_layer = search_layer(model_info, bottom)
                bottom_layer['output_name'].append(layer.name)
                dict['input_name'].append(bottom)
                dict['input_size'] = bottom_layer['output_size']


            pad = 0
            kernel_size = 0
            num_output = 0
            stride = 1
            group = 1

            if layer.convolution_param.pad :
                pad = layer.convolution_param.pad[0]
            if layer.convolution_param.kernel_size:
                kernel_size = layer.convolution_param.kernel_size[0]
            if layer.convolution_param.num_output:
                num_output = layer.convolution_param.num_output
            if layer.convolution_param.stride:
                stride = layer.convolution_param.stride[0]
            if layer.convolution_param.group:
                group = layer.convolution_param.group

            dict['kernel_size'] = kernel_size
            dict['stride'] = stride
            dict['pad'] = pad
            dict['group'] = group

            # dict['input_size'] = bottom_layer['output_size']
            dict['num_output'] = num_output
            dict['output_size'], dict['pad_in'], dict['pad_whole'] = \
                get_convolution_output_size(dict['input_size'], num_output, kernel_size, stride, pad)
            model_info.append(dict)
        elif layer.type == 'ReLU':
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['ReLU'] = True

        elif layer.type == 'Concat':
            bottom = layer.bottom
            input_size = (0,0,0)
            for one_bottom in bottom:
                dict['input_name'].append(one_bottom)
                bottom_layer = search_layer(model_info, one_bottom)
                bottom_layer['output_name'].append(layer.name)
                if input_size[0] == 0:
                    input_size = bottom_layer['output_size']
                else:
                    temp_list_input = list(input_size)
                    temp_list_input[0] += bottom_layer['output_size'][0]
                    input_size = tuple(temp_list_input)
            dict['input_size'] = input_size
            dict['output_size'] = input_size
            model_info.append(dict)

        elif layer.type == 'InnerProduct':
            # dict = props(layer)
            bottom = layer.bottom[0]
            dict['input_name'] = bottom

            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['output_name'].append(layer.name)
            dict['input_size'] = bottom_layer['output_size']
            dict['num_output'] = layer.inner_product_param.num_output

            dict['output_size'] = ( dict['num_output'],)
            model_info.append(dict)

        elif layer.type == 'Dropout':
            #dict_no = props(layer)
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['Dropout'] = True

        elif layer.type == "Eltwise":
            bottom = layer.bottom
            input_size = (0,0,0)
            for one_bottom in bottom:
                dict['input_name'].append(one_bottom)
                bottom_layer = search_layer(model_info, one_bottom)
                bottom_layer['output_name'].append(layer.name)

                if input_size[0] == 0:
                    input_size = bottom_layer['output_size']
                # else:
                #     input_size[0] += bottom_layer['output_size'][0]
            dict['input_size'] = input_size
            dict['output_size'] = input_size
            model_info.append(dict)

    return model_info

def convert_to_node(model_info):
    node_info = []
    for i in range(0,len(model_info)):
        temp_node = Node(**model_info[i])
        node_info.append(temp_node)
    return node_info



def printMessage(exitMessage_list, sizeMessage_list):
    result_info = "----------------------Result--------------------\n\n"

    if len(exitMessage_list) == 0 and len(sizeMessage_list) == 0:
        result_info += "Model A is totally the same with Model B.\n"
    if len(exitMessage_list) > 0:
        temp_layer_1 = exitMessage_list[0].a
        temp_layer_2 = exitMessage_list[0].b

def del_node(model_info):

    for i in model_info:
        if i['type'] in del_type_set:
            input_name = i['input_name'][0]
            output_name = i['output_name'][0]
            input_node = search_dict_node(model_info, input_name)
            output_node = search_dict_node(model_info, output_name)
            input_node['output_name'] = [output_node['name'],]
            output_node['input_name'] = [input_node['name'],]

            model_info.remove(i)

    return model_info


def get_caffe_model(net_file, input_size):

    caffe_root = '/home/zhangge/caffe/'
    # os.chdir(caffe_root)
    # net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
    # net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
    # net_file = caffe_root + 'models/default_resnet_50/train_val.prototxt'
    # net_file = caffe_root + 'models/default_vgg_19/train_val.prototxt'

    net = caffe_pb2.NetParameter()
    f = open(net_file, 'r')
    text_format.Merge(f.read(), net)
    phase = caffe_pb2.Phase.Value('TRAIN')
    input_size = (3, 224, 224)
    model_info = build_graph(net, input_size, phase)
    model_info = del_node(model_info)
    # node_info = convert_to_node(model_info)

    return model_info





caffe_root = '/home/zhangge/caffe/'
os.chdir(caffe_root)
# net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
# net_file = caffe_root + 'models/default_resnet_50/train_val.prototxt'
# net_file = caffe_root + 'models/default_vgg_19/train_val.prototxt'

# net_file = caffe_root + 'models/dummy_data/default_resnet_train_val_dummy.prototxt'

model_info = get_caffe_model(net_file, (3,224,224))
node_info = get_node_list(model_info)
print "1"
