import numpy as np

class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)

class ConvNode(Node):

    def set_conv_prop(self):

        #   inital the kernel_size, stride and pad for h and w
        stride_h = self.stride
        stride_w = self.stride

        kernel_h = self.kernel_size
        kernel_w = self.kernel_size

        pad_h = self.pad
        pad_w = self.pad

        input_h = self.input_size[1]
        input_w = self.input_size[2]

        output_h = int((float)(input_h + 2 * pad_h - kernel_h) / stride_h) + 1
        output_w = int((float)(input_w + 2 * pad_w - kernel_w) / stride_w) + 1

        pad_whole_h = input_h + 2 * pad_h
        pad_whole_w = input_w + 2 * pad_w

        self.pad_whole = [self.input_size[0], pad_whole_h, pad_whole_w]
        self.pad_in = [self.input_size[0], (output_h - 1) * self.stride + kernel_h, (output_w - 1) * self.stride + kernel_w]
        self.output_size = [self.num_output, output_h, output_w]

        # return output_size, pad_in, pad_whole

if __name__ == "__main__":
    test_dict = {'a': 1, 'b': 2, 'c':3}
    test_conv_dict =  {'name': u'conv1', 'output_size': [96, 54, 54], 'type': u'Convolution', 'num_output': 96, 'input_name': [u'data'], 'ReLU': True, 'pad_whole': [3, 224, 224], 'input_size': [3, 224, 224],'output_name': [u'norm1'], 'pad_in': [3, 223L, 223L]}
    one_node = Node(**test_conv_dict)
    print vars(one_node)