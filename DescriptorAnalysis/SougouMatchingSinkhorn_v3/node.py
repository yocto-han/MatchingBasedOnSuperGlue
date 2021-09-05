#-*-coding:utf-8-*-
import numpy as np


class Node:
    """"""
    def __init__(self, gid, coor, cls):
        self.gid = gid    #global id
        self.coor = coor #coordinate
        self.cls = cls   #class

    def __str__(self,):
        return "gid:{}, coor:{}, cls:{}".format(self.gid, self.coor, self.cls)

    def feature(self,):
        f = np.zeros((len(self.coor)+4))
        f[0:len(self.coor)] = self.coor
        f[len(self.coor)+self.cls-1]= 1.0
        return f
