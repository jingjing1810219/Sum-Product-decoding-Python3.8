import numpy as np
from numpy import math
from math import atanh, tanh
from test1 import cum_pro
from test1 import cum_add
from get_matrix import load_code
from numpy.random import randn, rand
import random

"""
H = np.array([[1, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 0],
              [1, 0, 0, 0, 1, 1],
              [0, 0, 1, 1, 0, 1]], dtype=np.float32)
H_S = H.shape

# numbers of row
K = H.shape[0]

# numbers of col
N = H.shape[1]

LLR = [-1.3863, 1.3863, -1.3863, 1.3863, -1.3863, -1.3863]
LLR = np.array(LLR)
"""

#################################################################################
# we want get the edges matrix to record message and figure the message's flow out


def H_edges_f(H):

    N = H.shape[1]
    K = H.shape[0]

    H_edges = np.zeros(H.shape, dtype=np.float32)
    k = 0
    for i in range(K):
        for j in range(N):
            if H[i, j] == 1:
                H_edges[i, j] = H[i, j] + k
                k += 1

    return H_edges


###################################################################################
# compute message from variable to check for initialize, and get the message matrix
def H_V_C_initial_f(H, H_edges, LLR):

    K = H.shape[0]
    N = H.shape[1]
    H_V_C_initial = np.zeros(H_edges.shape, dtype=np.float32)
    for i in range(K):
        for j in range(N):
            if H_edges[i, j] != 0:
                # input the channel message for each edges according their colum and so that we can get the the initial message only from channel
                H_V_C_initial[i, j] = LLR[j]

    return H_V_C_initial


#################################################################################
# use for compute the message from the c-v 
# 用来计算从v—c的信息
def H_C_V_f(H, H_V_C_initial, H_edges):

    K = H.shape[0] 
    N = H.shape[1]
    H_C_V = np.zeros(H_edges.shape, dtype=np.float32)
    for i in range(K):
        l = [] 
        for j in range(N):
            if H_V_C_initial[i, j] != 0:
                l.append(H_V_C_initial[i, j])
    # 遍历每一行，找到计算这条边所用的所有额外边，然后用来计算下一步的信息
    # travsal the each row and find the extrinc edge for compute the each message
        l_edges = []
        for jj in range(len(l)):
            l_edges.append(l[0:jj]+l[jj+1:])
    # 计算每一个从C—V的信息
    # compute the each message from c-v by extrinc messages
        l_C_V = []
        for k in range(len(l_edges)):
            # l_C_V.append(2*np.arctanh(cum_pro(l_edges[k])))
            l_C_V.append(cum_pro(l_edges[k]))

        l_C_V = list(2 * np.arctanh(l_C_V))
    # 把更新后的llr值放入到信息矩阵中
    # put the update message into the message matrix
        s = []
        for q in range(N):  # 先找到所有不为0元素的列下标
                            # find all the col index for each number is not 0
            if H_edges[i, q] != 0:
                s.append(q)
        for m in range(len(s)):  # 把我们的信息放入更新后的矩阵中
                                # put our message in to the message matrix
            H_C_V[i, s[m]] = l_C_V[m]

    return H_C_V


#################################################################################
# 算出V—C的信息
# compuate the message form v-c
def H_V_C_f(H, H_C_V, H_edges, LLR, no_self_mess=True):

    K = H.shape[0] 
    N = H.shape[1]
    H_V_C = np.zeros(H_edges.shape, dtype=np.float32)
    # 用来记录最后一次计算V的信息
    # record the terminal messge for each v
    final_mess = []

    # 先把每一列的信息做成一个列表
    # find the elem not 0
    for i in range(N):
        l = [] 
        for j in range(K):
            if H_edges[j, i] != 0:
                l.append(H_C_V[j, i])
    # 计算每一个边需要的其他边的信息（不算自己的信息）
    # get the extrin edge for each edge
        l_edges = []
        for jj in range(len(l)):
            l_edges.append(l[0:jj]+l[jj+1:])
    # 需要自己的边来计算
    # if we want terminal the iteration
        if not no_self_mess:
            l_edges = [] 
            for jj in range(len(l)):
                l_edges.append(l)
    # 根据边数矩阵列数来计算每一个V—C的信息
    # caculate the message by given message matrx for c-v    
        l_V_C = []
        for k in range(len(l_edges)):
            l_V_C.append(cum_add(l_edges[k]) + LLR[i])

    # 如果需要自己这条边的信息，那么我们就需要把它每一的信息保存下来
        # if we want get the final message then add all message from c -v
        if not no_self_mess:
            final_mess.append(l_V_C[0])

        # 找到每一列元素不为空的行数
        # find the row num if elem is not 0
        s = []
        for q in range(K):
            if H_edges[q, i] != 0: 
                s.append(q)
        # 根据行数把计算出来的信息输入到V—C信息矩阵
        # add the new message from v-c in to message matrix
        for m in range(len(s)):
            H_V_C[s[m], i] = l_V_C[m]
    # 如果是最后一次计算，则返回最后每一个V的总信息
    # if we want get the final message for each v
    if not no_self_mess:
        return np.array(final_mess)

    return H_V_C


##############################################################################
