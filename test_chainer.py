import alex
from chainer import Variable
import numpy as np

model = alex.Alex()

x = Variable(np.asarray([0,2], [1, -3]))

# for name, value in six.iteritems(model.params()):
#     print (name, value)