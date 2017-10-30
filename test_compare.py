from load_chainer_model import get_chainer_model
from rank_multi_port import get_node_list

conv_param_set = [
    'pad', 'kernel_size', 'stride', 'pad_in', 'pad_whole',
    'group',
]

com_param_set = [
    'output_size',
]

add_param_set = [
    'input_size',
    'output_size'
]

red_str = '\033[1;31m%s\033[0m'
layer_str = '%s (%s)%s: (%s)\n'

type_str = '%s (%s): %s\n'
class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)


class exitMessage(object):

    def __init__(self, index, node_a, node_b):
        self.index = index
        self.a = node_a
        self.b = node_b
        #self.param = param

    def build_log_str(self):

        self.log_a = 'a' + type_str % (str(self.index), self.a.name, self.a.type)
        self.log_b = 'b' + type_str % (str(self.index), self.b.name, self.b.type)

class sizeMessage(object):

    def __init__(self, index, node_a, node_b, type):
        self.index = index
        self.a = node_a
        self.b = node_b
        self.type = type
        #self.param = param

    def build_log_str(self):
        temp_str_a = ''
        temp_str_b = ''
        if self.type == 'Convolution' or self.type == 'Pooling':
            for one_param in conv_param_set:
                temp_a = getattr(self.a, one_param)
                temp_b = getattr(self.b, one_param)

                temp_a = one_param + '=' + str(temp_a)
                temp_b = one_param + '=' + str(temp_b)
                if temp_a != temp_b:
                    temp_a = red_str % (temp_a)
                    temp_b = red_str % (temp_b)

                temp_str_a += (temp_a + ', ')
                temp_str_b += (temp_b + ', ')
        elif self.type == 'Concat' or self.type == 'Eltwise' or self.type == 'InnerProduct':
            for one_param in com_param_set:
                temp_a = getattr(self.a, one_param)
                temp_b = getattr(self.b, one_param)

                temp_a = one_param + '=' + str(temp_a)
                temp_b = one_param + '=' + str(temp_b)
                if temp_a != temp_b:
                    temp_a = red_str % (temp_a)
                    temp_b = red_str % (temp_b)

                temp_str_a += (temp_a + ', ')
                temp_str_b += (temp_b + ', ')
        else:
            for one_param in add_param_set:
                temp_a = getattr(self.a, one_param)
                temp_b = getattr(self.b, one_param)

                temp_a = one_param + '=' + str(temp_a)
                temp_b = one_param + '=' + str(temp_b)
                if temp_a != temp_b:
                    temp_a = red_str % (temp_a)
                    temp_b = red_str % (temp_b)

                temp_str_a += (temp_a + ', ')
                temp_str_b += (temp_b + ', ')

        temp_str_a = 'a' + layer_str % (str(self.index), self.a.name, self.a.type, temp_str_a)
        temp_str_b = 'b' + layer_str % (str(self.index), self.b.name, self.b.type, temp_str_b)
        self.log_a = temp_str_a
        self.log_b = temp_str_b

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
                node_a.pad_whole == node_b.pad_whole and
                node_a.group == node_b.group)

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



# chainer_file = '/Users/Mac/zhanGGe/dl_model/train_googlenet.py'
chainer_file = '/Users/Mac/zhanGGe/dl_model/alex.py'
# class_name = 'GoogLeNet'
class_name = 'Alex'

model_info_1 = get_chainer_model(chainer_file, class_name)
node_info_1 = get_node_list(model_info_1)

# chainer_file = '/Users/Mac/zhanGGe/dl_model/train_googlenet.py'
chainer_file = '/Users/Mac/zhanGGe/dl_model/alex_new.py'
# class_name = 'GoogLeNet'
class_name = 'Alex'

model_info_2 = get_chainer_model(chainer_file, class_name)
node_info_2 = get_node_list(model_info_2)

exit_msg = []
size_msg = []

try:
    for i in range(0, len(node_info_1)):
        if not compare_type(node_info_1[i], node_info_2[i]):
            temp = exitMessage(i, node_info_1[i], node_info_2[i])
            temp.build_log_str()
            exit_msg.append(temp)
            break

        if not compare_param(node_info_1[i], node_info_2[i]):
            temp = sizeMessage(i, node_info_1[i], node_info_2[i], node_info_1[i].type)
            temp.build_log_str()
            size_msg.append(temp)

except:
    print "Some Error Happening in compare"


size_file_a = open('size_message_a.txt','w')
size_file_b = open('size_message_b.txt', 'w')
log_out_str = ''
exit_log_str = ''

if len(exit_msg) == 0:
    log_out_str += 'Topo Test Result: \n' \
                   '\tThe topolopy is SAME.'
else:
    log_out_str += 'Topo Test Result: \n' \
                   '\tThe topolopy is DIFFERENT. \n'
    log_out_str += 'The topo diffrence start from :\n' \
                   'node_a: ' + exit_msg[0].log_a+ '\n'\
                   'node_b: ' + exit_msg[0].log_b

if len(size_msg) == 0:
    log_out_str += 'Size Test Result: \n' \
                   '\tThe Size of model is SAME.'
else:
    log_out_str += 'Size Test Result: \n' \
                   '\tThe Size of model is DIFFERENT. \n' \
                   '\tThe detail: {\n'
size_log = 'The size different detail: \n'
for s_msg in size_msg:
    size_log += (s_msg.log_a + s_msg.log_b)
    size_log += '----------------------------------------------------------------------\n'

log_out_str += size_log

print log_out_str



