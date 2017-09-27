'''
dl_model:2017/08/17
autuor: ZheZhan
'''


import numpy as np
import sys,os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import math
from google.protobuf import text_format





class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)

def search_node(node_info, name):
    for i in range(0, len(node_info)):
        if node_info[i].name == name:
            return node_info[i]

class NodeLayer(object):
    def __init__(self, node, type):
        self.node = node
        self.type = type
    def get_output_name(self):
        output_name = self.node.output_name[0]
        temp_node = search_node(node_info, output_name)
        return temp_node

    def get_node_prop(self, **param):
        self.__dict__.update(param)

    def __cmp__(self, other):

        isOne = (self.type == other.type and self.input_size == other.input_size)
        if isOne == False:
            return False,'basic'
        else:
            if (self.type == 'Convolution' and other.type == 'Convolution') or \
                (self.type == 'Pooling' and other.type == 'Pooling') :
                return (isOne and self.kernel_size == other.kernel_size and self.stride == other.stride and \
                        self.pad == other.pad and self.pad_in == other.pad_in and self.pad_whole == self.pad_whole), 'pad'
            return True,'same'



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

            dict['kernel_size'] = kernel_size
            dict['stride'] = stride
            dict['pad'] = pad

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

            bottom_layer['output_name'] =[layer.name]

            dict['input_size'] = bottom_layer['output_size']
            dict['output_size'] = dict['input_size']
            dict['final'] = True
            model_info.append(dict)

        elif layer.type == "Eltwise":
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


def test_class_function():
    net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
    net = caffe_pb2.NetParameter()

    f = open(net_file, 'r')
    text_format.Merge(f.read(), net)
    phase = caffe_pb2.Phase.Value('TRAIN')

    input_size = [3, 224, 224]
    model_info = build_graph(net, input_size, phase)

    model_list = []
    thenode = model_info[0]

    while thenode['output_name'] != None:
        if len(thenode['output_name'] == 1) :
            model_list.append(thenode)

class lineNode(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.loss = False
        self.flag = False
    def get_line_length(self, node_info):
        thenode = self.start
        l = 0
        while thenode.name != self.end.name:
            l += 1
            if len(thenode.output_name) > 0:
                thenode = search_node(node_info, thenode.output_name[0])
            else:
                thenode = self.end # find the end node
                self.loss = True
        self.l = l
        return l

class BlockLayer(object):
    def __init__(self, input_name, type):
        self.input_name = input_name
        self.type =type
        self.start_node_set = []
        self.linenode_set = []

    def get_output_node(self):

        return self.output_node

    def build_block_prop(self, node_info):
        end_node = self.output_node
        for i in range(0,len(self.input_name)):
            if self.input_name[i] != end_node.name:
                temp_node = search_node(node_info, self.input_name[i])
            else:
                temp_node = None
            self.start_node_set.append(temp_node)

    def build_linenode_set(self, node_info):
        end_node = self.output_node
        for i in range(0, len(self.input_name)):

            start_node = search_node(node_info, self.input_name[i])

            temp = lineNode(start_node, end_node)
            temp.get_line_length(node_info)
            self.linenode_set.append(temp)

    def build_output_node(self, node_info):

        temp_node = search_node(node_info, self.input_name[len(self.input_name) - 1])
        while len(temp_node.input_name) == 1:
            if len(temp_node.output_name) > 1:
                exit('line_node find multi output_path')
            temp_node = search_node(node_info, temp_node.output_name[0])
        self.output_node = temp_node


def find_line_end(start_node):
    temp_node = start_node
    # find the end node having multi input path.
    while len(temp_node.input_name) == 1:
        if len(temp_node.output_name) > 1:
            exit('line_node find multi output_path')
        temp_node = search_node(node_info, temp_node.output_name[0])

    return temp_node

def convert_to_node(model_info):
    node_info = []
    for i in range(0,len(model_info)):
        temp_node = Node(**model_info[i])
        node_info.append(temp_node)
    return node_info

def convert_to_layer(node_info):
    layer_info = []
    input_node = node_info[0]
    temp_node = input_node
    while temp_node.output_name:
        temp_layer = NodeLayer(temp_node, 'node')
        layer_info.append(temp_layer)
        if len(temp_node.output_name) == 1:
            next_node = search_node(node_info, temp_node.output_name[0])
        else:
            temp_block_layer = BlockLayer(temp_node.output_name, 'block')
            temp_block_layer.build_output_node(node_info)
            temp_block_layer.build_block_prop(node_info)
            temp_block_layer.build_linenode_set(node_info)
            layer_info.append(temp_block_layer)

            next_node = temp_block_layer.get_output_node()

        temp_node = next_node
    temp_layer = NodeLayer(temp_node, 'node')
    layer_info.append(temp_layer)
    return layer_info


def test_convert_to_layer(node_info):
    layer_info = convert_to_layer(node_info)

    for i in layer_info:
        if i.type == 'node':
            print vars(i.node)
        elif i.type == 'block':
            print "block{\n"
            for j in i.linenode_set:
                print vars(j)
            print "}\n"

def compare_two_line(line_1, line_2, node_info):
    if line_1.l != line_2.l:
        return False,'line length is different.'
    line_1_temp_node = line_1.start
    line_2_temp_node = line_2.start
    for i in range(0,line_1.l):
        if line_1_temp_node != line_2_temp_node:
            return False, 'line node is diffrent.'
        line_1_temp_node = search_node(node_info, line_1_temp_node.output_name[0])
        line_2_temp_node = search_node(node_info, line_2_temp_node.output_name[0])
    return True

def compare_two_block(block_1, block_2):

    if len(block_1.linenode_set) != len(block_2.linenode_set):
        return False,'num of line topology'
    set_1 = block_1.linenode_set
    set_2 = block_2.linenode_set

    for i in range(0,len(set_1)):
        temp_line_1 = set_1[i]
        for j in range(0,len(set_2)):
            if compare_two_line(temp_line_1, set_2[j]):
                set_2.pop(j)
                set_1[i].issame = True
                break

    if len(set_2) == 0:
        return True
    else:
        return False



def compare_two_model(layer_one, layer_two):
    temp_layer_a = layer_one[0]
    temp_layer_b = layer_two[0]

    if len(layer_one) > len(layer_two):
        exit('length one is longer than length two')
    elif len(layer_one) < len(layer_two):
        exit('length one is shorter than length two')

    for i in range(0,len(layer_one)):
        temp_a = layer_one[i]
        temp_b = layer_two[i]

        if temp_a.type == temp_b.type:
            if temp_a.type == 'node':
                pass
            else:
                pass

        else:
            exit('The type is different.')


# if __name__ == "__main__":
caffe_root = '/home/zhangge/caffe/'
os.chdir(caffe_root)
# net_file = caffe_root + 'models/bvlc_alexnet/train_val.prototxt'
# net_file = caffe_root + 'models/bvlc_googlenet/train_val.prototxt'
net_file = caffe_root + 'models/default_resnet_50/train_val.prototxt'
# net_file = caffe_root + 'models/default_vgg_19/train_val.prototxt'
net = caffe_pb2.NetParameter()
f = open(net_file, 'r')
text_format.Merge(f.read(), net)
phase = caffe_pb2.Phase.Value('TRAIN')

input_size = [3, 224, 224]
model_info = build_graph(net, input_size, phase)
model_info_1 = model_info

node_info = convert_to_node(model_info)
# for i in range(0,len(node_info)):
#     print vars(node_info[i])

test_convert_to_layer(node_info)

    # test_final_model(model_info)
    # compare_model(model_info, model_info_1)
    # input_size = [3, 224, 224]

    # test_default_setting(net)
    # test_pool_pad()
    # test_convolution_pad()











