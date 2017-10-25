import sys

modulePath = '/home/zhangge/pymodel/resnet50.py'

import imp

foo = imp.load_source('aex', modulePath)

model = getattr(foo, 'alex')

print model