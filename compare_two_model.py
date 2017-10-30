import argparse

#from test_caffe import get_caffe_model
from load_chainer_model import get_chainer_model
from rank_multi_port import get_node_list

parser = argparse.ArgumentParser(
    description='This a model comparison tool')
parser.add_argument('--tool_1', '-t1', default='caffe',
                    help='the model 1 is what framework \
                    (caffe, chainer)')
parser.add_argument('--file_1', '-f1', help='model 1 file root')
parser.add_argument('--tool_2', '-t2', default='caffe',
                    help='the model 2 is what framework \
                    (caffe, chainer)')
parser.add_argument('--file_2', '-f2', help='model 1 file root')

parser.add_argument('--class_name_1', help='if the tool 1 is chainer, input the class name of model 1')
parser.add_argument('--class_name_2', help='if the tool 2 is chainer, input the class name of model 2')


args = parser.parse_args()


if args.tool_1 == 'caffe':
    if args.file_1:
        model_1 =  get_caffe_model(args.file_1)
    else:
        print("ERROR, please input the caffe file")
        exit()
elif args.tool_1 == 'chainer':
    class_name_1 = None
    if args.class_name_1:
        class_name_1 =  args.class_name_1
    else:
        print('ERROR, please input the chainer model class name')
        exit()

    if args.file_1:
        model_1 =  get_chainer_model(args.file_1, class_name_1)
    else:
        print('ERROR, please input the caffe file')
        exit()

if args.tool_2 == 'caffe':
    if args.file_2:
        model_2 = get_caffe_model(args.file_2)
    else:
        print("ERROR, please input the caffe file")
        exit()
elif args.tool_2 == 'chainer':
    class_name_2 = None
    if args.class_name_2:
        class_name_2 =  args.class_name_2
    else:
        print('ERROR, please input the chainer model class name')
        exit()

    if args.file_2:
        model_2 =  get_chainer_model(args.file_2, class_name_2)
    else:
        print('ERROR, please input the caffe file')
        exit()

# model_1 store the model_info of model 1
# model_2 store the model_info of model 2
node_info_1 = get_node_list(model_1)
node_info_2 = get_node_list(model_2)




