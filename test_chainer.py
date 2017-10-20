import alex
from chainer import Variable
import numpy as np
import chainer.computational_graph as c
import chainer.links as L
import googlenet
import resnet



def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = value
    return pr


#model = alex.Alex()
model = googlenet.GoogLeNet()
#model = resnet.ResNet()

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
f = L.Linear(3, 2)
y = f(x)
y.data

data = np.ndarray((128,3,model.insize,model.insize), dtype=np.float32)
data.fill(3333)

label = np.ndarray((128), dtype=np.int32)
label.fill(1)
x = np.asarray(data)
g = c.build_computational_graph(model.forward(x), remove_variable=False)
with open('./test_new', 'w') as o:
    o.write(g.dump())

# for one_node in g.nodes:
#     input = one_node.inputs[0]
#
#     output = one_node.outputs[0]
#     label = one_node




# for name, value in six.iteritems(model.params()):
#     print (name, value)