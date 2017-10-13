import chainer.links as L
import alex
import numpy as np
from chainer import function
from chainer import variable
from chainer.functions.array.get_item import GetItem
import googlenet

#model = alex.Alex()
model = googlenet.GoogLeNet()

data = np.ndarray((128,3,model.insize,model.insize), dtype=np.float32)
data.fill(3333)

label = np.ndarray((128), dtype=np.int32)
label.fill(1)
x = np.asarray(data)


functionnode = set()

outputs = model.forward(x)
output_set = set()
for o in outputs:
    if isinstance(o, variable.Variable):
        o = o.node
    while isinstance(o.creator, GetItem):
        o = o.creator.inputs[0]
    if o not in output_set:
        output_set.add(o)

def add_inputnode(one_var):
    temp_output_nodes = one_var
    while temp_output_nodes.creator and temp_output_nodes.creator not in functionnode:
        temp_func_node = temp_output_nodes.creator
        temp_func_node.input_nodes = set()
        temp_func_node.output_nodes = set()

        for one_input in temp_func_node.inputs:
            if one_input.creator:
                temp_func_node.input_nodes.add(one_input.creator)

        functionnode.add(temp_func_node)
        for one_var in temp_func_node.inputs:
            add_inputnode(one_var)


for one_out in output_set:
    add_inputnode(one_out)



# for one_out  in output_set:
#     temp_data_var = one_out
#     while temp_data_var.creator:
#         temp_func_node = temp_data_var.creator
#         temp_func_node.input_nodes = set()
#         temp_func_node.output_nodes = set()
#
#         temp_func_node.input_nodes.add(temp_func_node.inputs[0].creator)
#
#         functionnode.add(temp_func_node)
#
#         temp_data_var = temp_func_node.inputs[0]

for one_node in functionnode:
    temp_output_nodes = set()
    for temp_node in functionnode:
        if one_node in temp_node.input_nodes:
            temp_output_nodes.add(temp_node)
    one_node.output_nodes = temp_output_nodes

pass


for one_func_node in functionnode:
    if one_func_node.rank == 0:
        print one_func_node

# for o in model.forward(x):
#     if isinstance(o, variable.Variable):
#         nodes.add(o)
#         creator = o.creator
#         if creator is not None and creator not in functionnode:
#             functionnode.add(creator)
#
# for o in functionnode:
#     variablenode.add(o.inputs)
#
# for one_node in nodes:
#     if isinstance(one_node, variable.VariableNode):
#         variablenode.add(one_node)
#     elif isinstance(one_node, function.Function):
#         functionnode.add(one_node)
#
# pass


