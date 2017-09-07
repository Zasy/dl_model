'''
dl_model:2017/08/17
autuor: ZhaZha
'''


import numpy as np
import sys,os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import math
from google.protobuf import text_format



class lineNode(object):

    def __init__(self, start, starttype):
        self.start = start
        self.starttype = starttype

    def get_line_length(self):
        thenode = self.start
        self.l = 1
        while len(thenode['output_name']) == 1:
            self.l += 1
            thenode = thenode['output_name'][0]

class BlockNode(object):

    def __init__(self):
        self.line_list = []

    def set_before_node(self, node):
        self.before_node = node

    def add_line_node(self, line_node):
        self.line_list.append(line_node)

    def __len__(self):
        return len(self.line_list)


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

    pad_whole = [input_size[0], pad_whole_h, pad_whole_w]
    pad_in = [input_size[0], (output_h-1)*stride + kernel_h, (output_w-1)*stride +kernel_w]


    output_size = [input_size[0], output_h, output_w]

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

    pad_whole = [input_size[0], pad_whole_h, pad_whole_w]
    pad_in = [input_size[0], (output_h-1)*stride + kernel_h, (output_w-1)*stride +kernel_w]
    output_size = [num_output, output_h, output_w]

    return output_size, pad_in, pad_whole

def build_graph(net, input_size, phase):
    model_info = []
    for layer in net.layer:
        dict = {}
        dict['type'] = layer.type
        dict['name'] = layer.name
        dict['input_name'] = []
        dict['output_name'] = []
        if layer.type == 'Data':
            if layer.include[0].phase == phase:
                dict['locate'] = True
                dict['input_size'] = input_size
                dict['output_size'] = input_size
                model_info.append(dict)
        elif layer.type == 'Input':
            dict['locate'] = True
            dict['output_size'] = input_size
            model_info.append(dict)


        elif layer.type == 'Pooling':
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
            dict['output_size'],dict['pad_in'],dict['pad_whole'] \
                = get_pooing_output_size(dict['input_size'], kernel_size, stride, pad)
            model_info.append(dict)


        elif layer.type == 'LRN':

            bottom = layer.bottom[0]
            dict['input_name'].append(bottom)
            bottom_layer = search_layer(model_info, bottom)
            if bottom_layer.has_key('output_name'):
                bottom_layer['output_name'].append(layer.name)
            else:
                bottom_layer['output_name'] =[layer.name]
            # print bottom_layer['output_name']
            dict['input_size'] = bottom_layer['output_size']
            dict['output_size'] = dict['input_size']
            model_info.append(dict)


        elif layer.type == 'Convolution':

            bottom = layer.bottom[0]
            dict['input_name'].append(bottom)
            bottom_layer = search_layer(model_info, bottom)

            bottom_layer['output_name'].append(layer.name)



            pad = 0
            kernel_size = 0
            num_output = 0
            stride = 1
            if layer.convolution_param.pad :
                pad = layer.convolution_param.pad[0]
            if layer.convolution_param.kernel_size:
                kernel_size = layer.convolution_param.kernel_size[0]
            if layer.convolution_param.num_output:
                num_output = layer.convolution_param.num_output
            if layer.convolution_param.stride:
                stride = layer.convolution_param.stride[0]

            dict['input_size'] = bottom_layer['output_size']
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


            input_size = [0,0,0]
            for one_bottom in bottom:
                dict['input_name'].append(one_bottom)
                bottom_layer = search_layer(model_info, one_bottom)

                bottom_layer['output_name'].append(layer.name)


                if input_size[0] == 0:
                    input_size = bottom_layer['output_size']
                else:
                    input_size[0] += bottom_layer['output_size'][0]
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

            dict['output_size'] = [0,0]
            dict['output_size'][0] = dict['input_size'][0]
            dict['output_size'][1] = dict['num_output']
            model_info.append(dict)

        elif layer.type == 'Dropout':
            #dict_no = props(layer)
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)
            bottom_layer['Dropout'] = True

        elif layer.type == "Softmax":
            bottom = layer.bottom[0]
            bottom_layer = search_layer(model_info, bottom)

            bottom_layer['output_name'].append(layer.name)

            bottom_layer['output_name'] =[layer.anme]

            dict['input_size'] = bottom_layer['output_size']
            dict['output_size'] = dict['input_size']
            dict['final'] = True
            model_info.append(dict)

    return model_info

def sort_outputname(output):
    pass

def group_node(model_info):
    pass

def compare_two_node(a, b):
    print("===========================================================")
    print("Start to compare two node")
    print(a["type"],b["type"])
    if a['type'] == b['type'] :
        a_output_name = a['output_name']
        b_output_name = b['output_name']


        print a_output_name
        print b_output_name
        a_output_name = sorted(a_output_name)
        b_output_name = sorted(b_output_name)

        print("======================================================")
        print a_output_name
        print b_output_name



def compare_model(model_a, model_b):

    for i in range(0,len(model_a), 1):
        compare_two_node(model_a[i], model_b[i])




def test_final_model(model_info):

    for layer in model_info:
    #     print("type: %s\tinput_size: %r\toutput_size: %r", layer['type'], layer['input_size'], layer['output_size'])
        print layer

def test_default_setting(net):

    for layer in net.layer:
        if layer.type == "Convolution":
            pad = 0
            kernel_size = 0
            num_output = 0
            stride = 1
            if layer.convolution_param.pad :
                pad = layer.convolution_param.pad[0]
            if layer.convolution_param.kernel_size:
                kernel_size = layer.convolution_param.kernel_size[0]
            if layer.convolution_param.num_output:
                num_output = layer.convolution_param.num_output
            if layer.convolution_param.stride:
                stride = layer.convolution_param.stride[0]
            # print num_output
            # print stride
            # print kernel_size
            # print pad
        elif layer.type == "Pooling":
            kernel_size = 0
            stride = 1
            pad = 0
            if layer.pooling_param.kernel_size:
                kernel_size = layer.pooling_param.kernel_size
            if layer.pooling_param.pad:
                pad = layer.pooling_param.pad
            if layer.pooling_param.stride:
                stride = layer.pooling_param.stride
            # print kernel_size
            # print pad
            # print stride

def test_pool_pad():
    pad = 0
    kernel_size = 3
    stride = 2
    input_size = [64, 8, 8]

    print 'input_size: %r' % input_size
    print 'kernel_size: %d \npad: %d\nstride: %d' % (kernel_size, pad, stride)
    output_size, pad_in, pad_whole = get_pooing_output_size(input_size, kernel_size, stride, pad)
    print 'output_size: %r \npad_in: %r\npad_whole: %r\n' % (output_size, pad_in, pad_whole)

def test_convolution_pad():

    pad = 0
    kernel_size = 3
    stride = 2
    input_size = [64, 8, 8]
    num_output = 128

    print 'input_size: %r' % input_size
    print 'kernel_size: %d \npad: %d\nstride: %d' % (kernel_size, pad, stride)
    output_size, pad_in, pad_whole = get_convolution_output_size(input_size, num_output, kernel_size, stride, pad)
    print 'output_size: %r \npad_in: %r\npad_whole: %r\n' % (output_size, pad_in, pad_whole)



if __name__ == "__main__":
    caffe_root = '/home/zhangge/caffe/'
    os.chdir(caffe_root)
    # net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
    net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
    net = caffe_pb2.NetParameter()
    f = open(net_file, 'r')
    text_format.Merge(f.read(), net)

    phase = caffe_pb2.Phase.Value('TRAIN')

    input_size = [3, 224, 224]
    model_info = build_graph(net, input_size, phase)
    model_info_1 = model_info
    # test_final_model(model_info)

    compare_model(model_info, model_info_1)
    # input_size = [3, 224, 224]

    # test_default_setting(net)
    # test_pool_pad()
    # test_convolution_pad()











