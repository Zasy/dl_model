#!/usr/bin/env python
# coding=utf-8

### this is a tool that conduct a full computation graph,like tensorBoard
### from saved checkpoint file.

import tensorflow as tf
import re

# list of unwanted layers
del_type_set=['Relu','BiasAdd','BatchNorm','Pad','Dropout']

def del_BiasAdd_layer(inputs_name,d,net):
    if 'BiasAdd' in inputs_name :
        d['input_name'] = net[-1]['name']
    else:
        d['input_name'] = inputs_name

# tensorflow channel number is in the last dimension,but channel number is in the first demension for caffe
# so we need to change
def transpose_output_size(output_size):
    out = output_size.pop(-1)
    output_size.insert(0,out)
    return output_size

def get_input_size(op):
    input_size=op.inputs[0]._shape_as_list()[1:4]
    input_size=transpose_output_size(input_size)
    return tuple(input_size)

def get_input_name(op):
    in_name=[]
    for i in op.inputs:
        inputs_name=str(i)
        inputs_name = inputs_name.strip().split('"')[1]
        inputs_name = inputs_name.strip().split(':')[0]
        in_name.append(inputs_name)
    return in_name


def get_pad_size(j,d):
    if j.node_def.attr['padding'].s=='SAME':
        d['pad_whole']=(d['input_size'][0],(int(d['output_size'][1])-1)*d['stride']+d['kernel_size'],\
                                (int(d['output_size'][1])-1)*d['stride']+d['kernel_size'])
        d['pad_in']=d['pad_whole']
        d['pad']=((d['output_size'][1]-1)*d['stride']+d['kernel_size']-d['input_size'][1])/2
        d['ph']=int(d['pad_whole'][1])-int(d['input_size'][1])
        d['pw']=int(d['pad_whole'][2])-int(d['input_size'][2])
    if j.node_def.attr['padding'].s=='VALID':
        pad_whole=j.inputs[0]._shape_as_list()[1:4]
        d['pad_whole']=tuple(transpose_output_size(pad_whole))
        d['pad_in']=(d['input_size'][0],(int(d['output_size'][1])-1)*d['stride']+d['kernel_size'],\
                    (int(d['output_size'][1])-1)*d['stride']+d['kernel_size'])
        d['pad']=0
        d['ph']=0
        d['pw']=0

# search some previous or next nodes according to name
def search_dict_node(model_info, name,is_input=True):
    output_node_set=[]
    for i in range(0, len(model_info)):
        if is_input:
            if name == [model_info[i]['name']]:
                return model_info[i]
        else:
            if name in model_info[i]['input_name']:
                output_node_set.append(model_info[i])
    return output_node_set

def search_layer(net_layers, dst_name):
    i = 0
    for i in range(len(net_layers)):
        if net_layers[i]['name'] == dst_name:
            break
    return net_layers[i]

# Remove some unwanted layers
def del_node(model_info):
    for del_type in del_type_set:
        for i in model_info:
            if i['type'] == del_type:
                input_name = i['input_name']
                name = i['name']
                input_node = search_dict_node(model_info, input_name,is_input=True)
                output_node_set = search_dict_node(model_info, name,is_input=False)
                if output_node_set == None:
                   model_info.remove(i)
                   continue
                for output_node in output_node_set:
                    for j in range(0,len(output_node['input_name'])):
                       if  name in output_node['input_name'][j]:
                           output_node['input_name'][j]=input_node['name']
                model_info.remove(i)
    return model_info

# according to the op connection in the graph, we can add output name for every layer.
def add_output_name(model):
    for layer in model:
        out_name=[]
        for next_layer in model:
            if layer['name'] in next_layer['input_name']:
                out_name.append(next_layer['name'])
        layer['output_name']=out_name
    return model

# for tensorflow, add ops are too much to find the desired add op (eltwise in caffe)in the residual model
# so this function is used to find it
def is_Eltwise(inputs1_name,inputs2_name,net):
    all_input_name=[]
    for layer in net:
        all_input_name.append(layer['name'])
    if inputs1_name in all_input_name and inputs2_name in all_input_name:
        return True
    else:
        return False

def get_full_model(meta_path,ckpt_path):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        net=[]
        is_first_op_in_bn=True
        is_first_op_in_dropout=True
        is_data_input=True
        d_BN={}
        d_drop={}
        is_first_dropout=True
        for j in tf.get_default_graph().get_operations():
            #if j.type == 'Conv2D' or j.type == 'Relu' or j.type == 'MaxPool' \
            #            or j.type=='BiasAdd' or j.type=='LRN' or j.type=='MatMul':
            #    print (j.node_def)
            #    print ('----------------')
            # if 'dropout' in j.name:
            #     if is_first_dropout:
            #         print (j.node_def)

            d = {}
            if j.type == 'ConcatV2':
                d['type']='Concat'
                d['name']=j.name
                ##-------input name ----------------
                i=0
                in_name=[]
                in_size=[]
                while not 'axis' in str(j.inputs[i]):
                    inputs_name = str(j.inputs[i])
                    inputs_name = inputs_name.strip().split('"')[1]
                    inputs_name = inputs_name.strip().split(':')[0]
                    input_size = j.inputs[i]._shape_as_list()[1:4]
                    input_size = tuple(transpose_output_size(input_size))
                    in_name.append(inputs_name)
                    in_size.append(input_size)
                    i+=1

                d['input_name']=in_name
                d['input_size']=in_size
                output_size=j.outputs[0]._shape_as_list()[1:4]
                output_size=transpose_output_size(output_size)
                d['output_size']=tuple(output_size)
                net.append(d)
            if j.type == 'Conv2D':
                d['type']='Convolution'
                d['name']=j.name
                d['group']=1
                ## note: in some models, we can find that pad op is done outside of conv,
                ## so we need check it and change the conv layer.
                if net:
                    if net[-1]['type']=='Pad':
                        d['input_size']=net[-1]['input_size']
                    else:
                        d['input_size']=get_input_size(j)
                else:
                    d['input_size']=get_input_size(j)

                kernel_size=j.inputs[1]._shape_as_list()[0]
                d['kh']=kernel_size
                d['kw']=j.inputs[1]._shape_as_list()[1]
                d['kernel_size']=int(kernel_size)
                if is_data_input:
                    d['input_name']=[]
                    is_data_input=False
                else:
                    inputs_name=[get_input_name(j)[0]]
                    if 'dropout' in inputs_name or 'BiasAdd' in inputs_name:
                        d['input_name'] = net[-3]['name']
                    else:
                        d['input_name'] = inputs_name
                #------------------------------------------------
                output_size=j.outputs[0]._shape_as_list()[1:4]
                output_size=transpose_output_size(output_size)
                d['output_size']=tuple(output_size)
                d['num_output']=int(output_size[0])

                a = j.node_def.attr['strides']
                m = re.findall(r'(\w*[0-9]+)\w*',str(a))
                b = [int(x) for x in m]
                d['stride']=int(b[1])
                d['sh']=int(b[1])
                d['sw']=int(b[2])
                #----------padding-------------------------------
                if net:
                    if net[-1]['type']=='Pad':
                        d['pad_whole']=tuple(net[-1]['output_size'])
                        d['pad_in']=tuple(net[-1]['output_size'])
                        d['pad']=(int(d['pad_whole'][1])-int(d['input_size'][1]))/2
                        d['ph']=int(d['pad_whole'][0])-int(d['input_size'][1])
                        d['pw']=int(d['pad_whole'][1])-int(d['input_size'][2])
                    else:	
                        get_pad_size(j,d)
                else:
                    get_pad_size(j,d)
                net.append(d)
            if j.type == 'MaxPool' or j.type == 'AvgPool':
                if j.type == 'MaxPool':
                    d['type']='MaxPooling'
                if j.type == 'AvgPool':
                    d['type']='AveragePooling'
                d['name']=j.name
                # get inputs

                inputs_name = get_input_name(j)
                d['input_name']=inputs_name
                d['input_size'] =get_input_size(j)
                # get strides
                a = j.node_def.attr['strides']
                m = re.findall(r'(\w*[0-9]+)\w*',str(a))
                b = [int(x) for x in m]
                d['stride']=int(b[1])
                # get kernel_size
                a = j.node_def.attr['ksize']
                m = re.findall(r'(\w*[0-9]+)\w*',str(a))
                b = [int(x) for x in m]
                d['kernel_size']=int(b[1])
                output_size=j.outputs[0]._shape_as_list()[1:4]
                output_size=transpose_output_size(output_size)
                d['output_size']=tuple(output_size)
                #----------padding----------------------------
                get_pad_size(j,d)
                net.append(d)
            if j.type == 'Relu':
                d['type'] = 'Relu'
                d['name'] = j.name
                inputs_name = get_input_name(j)
                d['input_name']=inputs_name

                net.append(d)

            if j.type == 'LRN':
                d['type']=j.type
                d['name']=j.name
                # get inputs

                d['input_name'] = get_input_name(j)

                del_BiasAdd_layer(inputs_name,d,net)
                net.append(d)

            if j.type == 'BiasAdd':
                d['type']=j.type
                d['name']=j.name
                inputs_name = get_input_name(j)
                d['input_name']=[inputs_name[0]]
                net.append(d)

            if j.type == 'MatMul':
                d['type']='InnerProduct'
                d['name']=j.name

                
                inputs_name = [get_input_name(j)[0]]
                d['input_name']=[inputs_name]
                if 'Reshape' in inputs_name:
                    d['input_name'] = net[-1]['name']
                else:
                    d['input_name'] = inputs_name
                input_size=j.inputs[0]._shape_as_list()[1]
                d['input_size']=int(input_size)
                output_size=j.outputs[0]._shape_as_list()[1]
                d['output_size']=int(output_size)

                net.append(d)

            in_name_ = get_input_name(j)

            ## ---------get BN layer-----------------------------
            if 'batchnorm' in j.name:
                d_BN['type']='BatchNorm'
                d_BN['name']=j.name
                if is_first_op_in_bn:
                    d_BN['input_name']=[net[-1]['name']]
                    is_first_op_in_bn=False

            if 'batchnorm/add_1' in j.name:
                net.append(d_BN)
                d_BN={}
                is_first_op_in_bn=True
            ## ------------------------------------------

            ## get dropout layer-----------------------------
            if 'dropout' in j.name:
                d_drop['type']='Dropout'
                d_drop['name']=j.name
                if is_first_op_in_dropout:
                    d_drop['input_name']=[net[-1]['name']]
                    is_first_op_in_dropout=False

            if 'dropout/mul' in j.name:
                net.append(d_drop)
                d_drop={}
                is_first_op_in_dropout=True
            ## ------------------------------------------
            if j.type == 'Add':
                is_Eltwise_=is_Eltwise(in_name_[0],in_name_[1],net)
                if not is_Eltwise_:
                    continue
                else:
                    in_size=[]
                    input_size1 = j.inputs[0]._shape_as_list()[1:4]
                    input_size1 = tuple(transpose_output_size(input_size1))
                    in_size.append(input_size1)
                    input_size2 = j.inputs[1]._shape_as_list()[1:4]
                    input_size2 = tuple(transpose_output_size(input_size2))
                    in_size.append(input_size2)
                    d['type']='Eltwise'
                    d['name']=j.name
                    d['input_name']=in_name_
                    d['input_size']=in_size
                    output_size=j.outputs[0]._shape_as_list()[1:4]
                    output_size=transpose_output_size(output_size)
                    d['output_size']=tuple(output_size)
                net.append(d)
            if j.type == 'Pad':
                inputs_name=[in_name_[0]]
                d['input_size'] = get_input_size(j)
                output_size=j.outputs[0]._shape_as_list()[1:4]
                output_size=transpose_output_size(output_size)
                d['output_size']=output_size
                d['input_name'] = inputs_name
                d['type'] = j.type
                d['name'] = j.name
                net.append(d)
    return net

def get_model_with_desired_layer(meta_path,ckpt_path):
    net= get_full_model(meta_path,ckpt_path)
    net = del_node(net)
    net = add_output_name(net)
    return net

# net = get_model_with_desired_layer('./inception_v1/inception_v1/test_inception_v1.meta','./inception_v1/inception_v1/')
# for layer in net:
#     print (layer['input_name'],layer['name'],layer['type'])
