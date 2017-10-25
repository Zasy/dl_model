def search_node(node_info, name):
    for i in range(0, len(node_info)):
        if node_info[i].name == name:
            return node_info[i]

def build_nodes(node_info):

    for one_node in node_info:
        input_nodes = []
        output_nodes = []
        for one_input in one_node.input_name:
            tmp = search_node(node_info, one_input)
            input_nodes.append(tmp)

        for one_output in one_node.output_name:
            tmp = search_node(node_info, one_output)
            output_nodes.append(tmp)

        one_node.input_nodes = input_nodes
        one_node.output_nodes = output_nodes

    return node_info



# (Convolution -> 2 , kernel_size)
# (AveragePooling  -> 1, kernel_size)
# (MaxPooling -> 3, kernel_size)
# (InnerProduct -> 4, kernel_size)

#code format: (type_code, size_code)
def get_outset_code(out_set):
    the_code_set  = []
    tmp = (0,0)
    temp_node = out_set
    while len(temp_node.input_nodes) < 2 :
        if temp_node.type == 'Convolution':
            tmp = (2, int(temp_node.kernel_size))

        elif temp_node.type == 'AveragePooling':
            tmp = (1, int(temp_node.kernel_size))
        elif temp_node.type == 'MaxPooling':
            tmp = (3, int(temp_node.kernel_size))
        elif temp_node.type == 'InnerProduct':
            tmp = (4, int(temp_node.output_size[0]))

        the_code_set.append(tmp)

        if len(temp_node.output_nodes) > 0 :
            temp_node = temp_node.output_nodes[0]
        else:
            break

    return the_code_set

# compare two node
# 0 means code_a = code_b
# 1 means code_a > code_b
# 2 means code_a < code_b

def compare_two_code(code_a, code_b):
    if code_a[0]  < code_b[0]:
        return 2
    elif code_a[0] > code_b[0]:
        return 1
    else:
        if code_a[1] < code_b[1]:
            return 2
        elif code_a[1] > code_b[1]:
            return 1
        else:
            return 0
def compare_two_set(set_a, set_b):

    while True:
        if len(set_a) == 0 and len(set_b) == 0:
            return 0
        elif len(set_a) == 0 and len(set_b) != 0:
            return 2
        elif len(set_a) != 0 and len(set_b) == 0:
            return 1

        tmp_a = set_a[0]
        tmp_b = set_b[0]

        the_temp_result = compare_two_code(tmp_a, tmp_b)

        if the_temp_result  == 1:
            return 1
        elif the_temp_result == 2:
            return 2
        else:
            set_a.remove(tmp_a)
            set_b.remove(tmp_b)

def change_nodes_list(node_info):

    for one_node in node_info:
        if len(one_node.output_nodes) > 1:
            for i in range(0,len(one_node.output_nodes)-1):
                for j in range(0, len(one_node.output_nodes)-1-i):
                    tmp_a = get_outset_code(one_node.output_nodes[j])
                    tmp_b = get_outset_code(one_node.output_nodes[j+1])

                    if compare_two_set(tmp_a, tmp_b) == 1:
                        tmp = one_node.output_nodes[j]
                        one_node.output_nodes[j] = one_node.output_nodes[j+1]
                        one_node.output_nodes[j+1] = tmp
    return node_info


