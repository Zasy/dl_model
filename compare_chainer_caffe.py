import os


class Node(object):

    def __init__(self, **param):
        self.__dict__.update(param)