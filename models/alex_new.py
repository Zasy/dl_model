import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):
    insize = 224

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  64, 11, stride=4, pad=2),
            conv2=L.Convolution2D(64, 192,  5, pad=2),
            conv3=L.Convolution2D(192, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 256,  3, pad=1),
            conv5=L.Convolution2D(256, 256,  3, pad=1),
            fc6=L.Linear(256 * 6 * 6, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 2000),
        )

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2, cover_all=False)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accurary': F.accuracy(h,t)}, self)

        return loss
