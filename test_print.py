# from tabulate import tabulate
from texttable import Texttable
from termcolor import colored

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


a = [231,323,242]
b = [321,132,112]

# def conv_format(node):


test_s1 = '(name): %s\n' % ('name') + \
          '(size): in=%s, out=%s\n' % ('*'.join(str(e) for e in a), '*'.join(str(e) for e in b) )+ \
          '(%s): kernel_size=%d, stride=%d, pad=%d\n' % ('conv1', 2,1,2)
test_s2 = '(conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)'


# table = [[test_s1, test_s2]]
# print tabulate(table, headers=["Model_A","Model_B"], tablefmt='orgtbl')

table = Texttable()
table.set_deco(Texttable.HEADER)

Header = ["Model_A","Model_B"]
# table.set_cols_dtype(['t',  # text
#                       'f',  # float (decimal)
#                       'e',  # float (exponent)
#                       'i',  # integer
#                       'a'])  # automatic

table.set_cols_dtype(['t', 't'])
table.set_cols_align(['l', 'l'])
table.set_cols_width([60,60])
table.add_rows([Header,[colored(test_s1,'red'), colored(test_s2,'green')]])

print table.draw()
