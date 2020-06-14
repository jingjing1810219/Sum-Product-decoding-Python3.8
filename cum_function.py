import numpy as np
import math


def cum_pro(l):
    if len(l) == 0:
        return 0 
    elif len(l) == 1:
        s = np.tanh(l[0]/2)
        #return np.tanh(l[0]/2) 
        if s == 1:
            return 0.9999999999999 * s
        else:
            return s 
    else:
        m = np.tanh((l[-1])/2)
        if m == 1:
            m = m * 0.999999999999
        else:
            m = m
        return cum_pro(l[-2::-1]) * m

def cum_add(l):
    if len(l) == 0:
        return 0
    if len(l) == 1:
        return l[0]+0
    else:
        return cum_add(l[-2::-1]) + l[-1]


s = 2*np.arctanh(cum_pro([-1.3863, 1.3863]))

l = [100]


l_new = []
for i in range(len(l)):
    l_new.append(l[0:i]+l[i+1:])

a = np.arr


l_new2 = []
for jj in range(len(l_new)):
    l_new2.append(cum_pro(l_new[jj]))

