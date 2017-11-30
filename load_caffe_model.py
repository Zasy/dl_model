'''
dl_model:2017/08/17
autuor: ZheZhan
'''

import numpy as np
import sys,os
import math
from google.protobuf import text_format
# from tabulate import tabulategit


del_type_set = ['LRN', 'ReLU', 'Scale', 'BatchNorm', 'Dropout', 'Data', \
                'DummyData', 'Softmax', 'SoftmaxWithLoss', 'Accuracy','Flatten']



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
def get_pooing_output_size(input_size, kh, kw, sh, sw, ph, pw):

    stride_h = sh
    stride_w = sw

    kernel_h = kh
    kernel_w = kw

    pad_h = ph
    pad_w = pw

    input_h = input_size[1]
    input_w = input_size[2]

    # print((float)(input_h + 2*pad_h - kernel_h)/stride)
    output_h = int(math.ceil((float)(input_h + 2*pad_h - kernel_h)/stride_h) + 1)
    output_w = int(math.ceil((float)(input_w + 2*pad_w - kernel_w)/stride_w) + 1)
    if pad_h or pad_w:
        if (output_h - 1)*stride_h >= (input_h + pad_h):
            output_h-=1
        if (output_w - 1)*stride_w >= (input_w + pad_w):
            output_w -= 1
    pad_whole_h = input_h + 2*pad_h
    pad_whole_w = input_w + 2*pad_w

    pad_in_h = (output_h-1)*stride_h + kernel_h
    pad_in_w = (output_w-1)*stride_w + kernel_w

    if pad_in_h > pad_whole_h :pad_whole_h = pad_in_h
    if pad_in_w> pad_whole_w: pad_whole_w = pad_in_h

    pad_whole = (input_size[0], pad_whole_h, pad_whole_w)
    pad_in = (input_size[0], (output_h-1)*stride_h + kernel_h, (output_w-1)*stride_w +kernel_w)


    output_size = (input_size[0], output_h, output_w)

    return output_size, pad_in, pad_whole

# get the convolution detail
# return:
#   output_size: the output_size of convolution_size
#   pad_in: the data_size in the convolution op
#   pad_whole : the data_size after pad
def get_convolution_output_size(input_size, num_output, kh, kw, sh, sw, ph, pw):
    #   inital the kernel_size, stride and pad for h and w
    stride_h = sh
    stride_w = sw

    kernel_h = kh
    kernel_w = kw

    pad_h = ph
    pad_w = pw

    input_h = input_size[1]
    input_w = input_size[2]

    output_h = int((float)(input_h + 2*pad_h - kernel_h)/stride_h) + 1
    output_w = int((float)(input_w + 2*pad_w - kernel_w)/stride_w) + 1

    pad_whole_h = input_h + 2*pad_h
    pad_whole_w = input_w + 2*pad_w

    pad_whole = (input_size[0], int(pad_whole_h), int(pad_whole_w))
    pad_in = (input_size[0], int((output_h-1)*stride_h + kernel_h), int((output_w-1)*stride_w +kernel_w))
    output_size = (num_output, output_h, output_w)

    return output_size, pad_in, pad_whole

def build_node(net, phase):
    model_info = []

    Data_layer_name = None
    i = 0
    for layer in net.layer:
        dict = {}
        dict['type'] = layer.type
        dict['name'] = layer.name +'_' +str(i)
        dict['bottom'] = layer.bottom
        dict['top'] = layer.top
        dict['input_name'] = []
        dict['output_name'] = []
        dict['inplace_name'] = []
        dict['input_size'] = ()
        dict['output_size'] = ()
        dict['rank'] = 1
        if layer.type == 'DummyData':
            if layer.include[0].phase == phase:
                Data_layer_name = layer.name
                dict['rank'] = -1
        elif layer.type == 'Data':
            if layer.include[0].phase == phase:

                Data_layer_name = layer.name
                dict['rank'] = -1
        elif layer.type == 'Pooling':
            if layer.pooling_param.pool == 0:
                dict['type'] = 'MaxPooling'
            else:
                dict['type'] = 'AveragePooling'

            kernel_size = 0
            stride = 1
            pad = 0
            if layer.pooling_param.pad :
                pad = layer.pooling_param.pad
                pw = pad
                ph = pad
            else:
                if layer.pooling_param.pad_h:
                    ph = layer.pooling_param.pad_h
                else:
                    ph = pad

                if layer.pooling_param.pad_w:
                    pw = layer.pooling_param.pad_w
                else:
                    pw = pad
            if layer.pooling_param.kernel_size:
                kernel_size = layer.pooling_param.kernel_size
                kh = kernel_size
                kw = kernel_size
            else:
                if layer.pooling_param.kernel_h:
                    kh = layer.pooling_param.kernel_h
                if layer.pooling_param.kernel_w:
                    kw = layer.pooling_param.kernel_w
            if layer.pooling_param.stride:
                stride = layer.pooling_param.stride
                sh = stride
                sw = stride
            else:
                if layer.pooling_param.stride_h:
                    sh = layer.pooling_param.stride_h
                else:
                    sh = stride
                if layer.pooling_param.stride_w:
                    sw = layer.pooling_param.stride_w
                else:
                    sw = stride

            dict['kernel_size'] = kernel_size
            dict['stride'] = stride
            dict['pad'] = pad
            dict['kh'] = kh
            dict['kw'] = kw
            dict['sh'] = sh
            dict['sw'] = sw
            dict['pw'] = pw
            dict['ph'] = ph

        elif layer.type == 'Convolution':
            pad = 0
            kernel_size = 0
            num_output = 0
            stride = 1
            group = 1

            if layer.convolution_param.pad :
                pad = layer.convolution_param.pad[0]
                pw = pad
                ph = pad
            else:
                if layer.convolution_param.pad_h:
                    ph = layer.convolution_param.pad_h
                else:
                    ph = pad

                if layer.convolution_param.pad_w:
                    pw = layer.convolution_param.pad_w
                else:
                    pw = pad
            if layer.convolution_param.kernel_size:
                kernel_size = layer.convolution_param.kernel_size[0]
                kh = kernel_size
                kw = kernel_size
            else:
                if layer.convolution_param.kernel_h:
                    kh = layer.convolution_param.kernel_h
                if layer.convolution_param.kernel_w:
                    kw = layer.convolution_param.kernel_w

            if layer.convolution_param.num_output:
                num_output = layer.convolution_param.num_output
            if layer.convolution_param.stride:
                stride = layer.convolution_param.stride[0]
                sh = stride
                sw = stride
            else:
                if layer.convolution_param.stride_h:
                    sh = layer.convolution_param.stride_h
                else:
                    sh = stride
                if layer.convolution_param.stride_w:
                    sw = layer.convolution_param.stride_w
                else:
                    sw = stride
            if layer.convolution_param.group:
                group = layer.convolution_param.group

            dict['kernel_size'] = kernel_size
            dict['stride'] = stride
            dict['pad'] = pad
            dict['group'] = group
            dict['num_output'] = num_output
            dict['kh'] = kh
            dict['kw'] = kw
            dict['sh'] = sh
            dict['sw'] = sw
            dict['pw'] = pw
            dict['ph'] = ph



        elif layer.type == 'InnerProduct':
            dict['num_output'] = layer.inner_product_param.num_output

        model_info.append(dict)
        i += 1

    return model_info

def find_node(model_info, node_name, bottom_name):
    for one_in_node in model_info:
        if one_in_node['name'] != node_name:
            if bottom_name in one_in_node['top']:
                return one_in_node

def find_model_dict(layer_name, model_info):
    for one_model in model_info:
        if one_model['name'] == layer_name:
            return one_model

def add_inputs_outputs(model_info):

    for one_node in model_info:
        if one_node['top'] == one_node['bottom']:
            for one_bottom in one_node['bottom']:
                cor_node = find_node(model_info, one_node['name'], one_bottom)
                cor_node['inplace_name'].append(one_node['name'])
        else:
            for one_bottom in one_node['bottom']:
                cor_node = find_node(model_info, one_node['name'], one_bottom)
                if cor_node['top'] != cor_node['bottom']:
                    one_node['input_name'].append(cor_node['name'])
                    cor_node['output_name'].append(one_node['name'])

    return model_info

def insert_node(temp_node, one_inplace, model_info):
    temp_output_name = temp_node['output_name']
    temp_node['output_name'] = [one_inplace['name'],]
    one_inplace['output_name'] = temp_output_name
    one_inplace['input_name'] = [temp_node['name'],]
    for one_output_name in temp_output_name:
        temp_one_out = find_model_dict(one_output_name, model_info)
        temp_one_out['input_name'].remove(temp_node['name'])
        temp_one_out['input_name'].append(one_inplace['name'])

    return model_info

def remove_inplace(model_info):
    for one_node in model_info:
        if len(one_node['inplace_name']) > 0:
            temp_node = one_node
            for one_inplace_name in one_node['inplace_name']:
                one_inplace = find_model_dict(one_inplace_name, model_info)

                model_info = insert_node(temp_node, one_inplace, model_info)

                temp_node = one_inplace

    return model_info


def build_variable_size(model_info, input_size):
    for one_node in model_info:
        if one_node['type'] == 'DummyData' or one_node['type'] == 'Data':
            one_node['input_size'] = input_size
            one_node['output_size'] = input_size
        elif one_node['type'] == 'Convolution':
            num_output = one_node['num_output']
            kw = one_node['kw']
            kh = one_node['kh']
            pw = one_node['pw']
            ph = one_node['ph']
            sh = one_node['sh']
            sw = one_node['sw']

            input_size = (0, 0, 0)
            for one_input_name in one_node['input_name']:
                tmp_node = find_model_dict(one_input_name, model_info)
                if input_size[0] == 0:
                    input_size = tmp_node['output_size']
            one_node['input_size'] = input_size
            one_node['output_size'], one_node['pad_in'], one_node['pad_whole'] = \
                get_convolution_output_size(input_size, num_output, kh, kw, sh, sw, ph, pw)

        elif one_node['type'] == 'MaxPooling' or one_node['type'] == 'AveragePooling':
            kernel_size = one_node['kernel_size']
            pad = one_node['pad']
            stride = one_node['stride']
            kw = one_node['kw']
            kh = one_node['kh']
            pw = one_node['pw']
            ph = one_node['ph']
            sh = one_node['sh']
            sw = one_node['sw']

            input_size = (0, 0, 0)
            for one_input_name in one_node['input_name']:
                tmp_node = find_model_dict(one_input_name, model_info)
                if input_size[0] == 0:
                    input_size = tmp_node['output_size']

            one_node['output_size'],one_node['pad_in'],one_node['pad_whole'] \
                = get_pooing_output_size(input_size, kh, kw, sh, sw, ph, pw)

        elif one_node['type'] == 'Concat':
            input_size = (0, 0, 0)
            for one_input_name in one_node['input_name']:
                tmp_node = find_model_dict(one_input_name, model_info)
                if input_size[0] == 0:
                    input_size = tmp_node['output_size']
                else:
                    temp_list_input = list(input_size)
                    temp_list_input[0] += tmp_node['output_size'][0]
                    input_size = tuple(temp_list_input)
            one_node['input_size'] = input_size
            one_node['output_size'] = input_size

        elif one_node['type'] == 'InnerProduct':
            input_size = 1
            one_input_name  = one_node['input_name'][0]
            one_input = find_model_dict(one_input_name, model_info)


            for i in one_input['output_size']:
                input_size *= i
            one_node['input_size'] = (input_size,)
            one_node['output_size'] = (one_node['num_output'],)

        elif one_node['type'] == "Eltwise":
            input_size = (0, 0, 0)
            for one_input_name in one_node['input_name']:
                tmp_node = find_model_dict(one_input_name, model_info)
                if input_size[0] == 0:
                    input_size = tmp_node['output_size']

            one_node['input_size'] = input_size
            one_node['output_size'] = input_size

        else:
            input_size = (0, 0, 0)
            for one_input_name in one_node['input_name']:
                tmp_node = find_model_dict(one_input_name, model_info)
                if input_size[0] == 0:
                    input_size = tmp_node['output_size']


            one_node['input_size'] = input_size
            one_node['output_size'] = input_size
    return model_info

# def convert_to_node(model_info):
#     node_info = []
#     for i in range(0,len(model_info)):
#         temp_node = Node(**model_info[i])
#         node_info.append(temp_node)
#     return node_info


def del_node(model_info):

    new_model_info = []
    for one_node in model_info:
        one_node['remove'] = False
        if one_node['type'] in del_type_set:
            if len(one_node['input_name']) > 0:
                input_node = find_model_dict(one_node['input_name'][0],model_info)
                input_node['output_name'].remove(one_node['name'])
                if len(one_node['output_name']) > 0:
                    for one_temp_output in one_node['output_name']:

                        output_node = find_model_dict(one_temp_output, model_info)
                        input_node['output_name'].append(output_node['name'])
                        output_node['input_name'].remove(one_node['name'])
                        output_node['input_name'].append(input_node['name'])

            one_node['remove'] = True

    for one_node in model_info:
        if one_node['remove'] == False:
            new_model_info.append(one_node)


    return new_model_info


def get_caffe_model(net_file):

    caffe_root = '/home/zhangge/caffe/'
    # os.chdir(caffe_root)
    # net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
    # net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
    # net_file = caffe_root + 'models/default_resnet_50/train_val.prototxt'
    # net_file = caffe_root + 'models/default_vgg_19/train_val.prototxt'


    import sys
    # setting the intel_caffe_root
    sys.path.append('/home/zhangge/caffe/python')
    import caffe
    import caffe.proto.caffe_pb2 as caffe_pb2

    net_model = caffe.Net(net_file, caffe.TRAIN)
    input_size = net_model.blobs['data'].data.shape[1:]
    net = caffe_pb2.NetParameter()
    f = open(net_file, 'r')
    text_format.Merge(f.read(), net)
    phase = caffe_pb2.Phase.Value('TRAIN')
    #input_size = (3, 224, 224)
    model_info = build_node(net, phase)
    model_info = add_inputs_outputs(model_info)
    model_info = remove_inplace(model_info)

    model_info = build_variable_size(model_info, input_size)
    model_info = del_node(model_info)

    return model_info





# caffe_root = '/home/zhangge/caffe/'
# os.chdir(caffe_root)
# # net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
# net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
# # net_file = caffe_root + 'models/default_resnet_50/train_val.prototxt'
# # net_file = caffe_root + 'models/default_vgg_19/train_val.prototxt'
# # net_file = caffe_root + 'models/dummy_data/default_resnet_train_val_dummy.prototxt'
# # net_file = caffe_root + 'models/googlenet_v3/train_val_mkl.prototxt'
#
# model_info = get_caffe_model(net_file)
#
# node_info = get_node_list(model_info)
