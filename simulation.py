import numpy as np
from get_matrix import load_code
from numpy.random import randn, rand
from numpy import math
import random
import SPA_function

random.seed(999)

# load the H and G for our decoding
#m = load_code("./code/BCH_15_11.alist", "./code/BCH_15_11.gmat")

m = load_code("./code/BCH_63_45.alist", "./code/BCH_63_45.gmat")
H = np.load("Parity.npy")
print(H)
G = np.load("Generate.npy")
G = G.T
# number of row for H
K = H.shape[0]
# num of colum for H
N = H.shape[1]
# length of codeword
s = G.shape[0]
# get the edge matrix for H
E = SPA_function.H_edges_f(H)

# set the snr range
EbN0dB_range = range(1, 9)
itr = len(EbN0dB_range)
# code rate for given code
r = (N - K) / N
# num of frame for decode
num_fram = 400
BER = []
for n in range(0, itr):
    # caculate the sigma for channel by give EbN0
    EbN0dB = EbN0dB_range[n]
    EbN0 = 10.0 ** (EbN0dB / 10.0)
    m = 2 * r * EbN0
    n = 1 / m
    noise_std = np.sqrt(n)
    # set the varable for results
    fram_count = 0
    bit_error = 0
    frame_error = 0
    # starting decoding for each frame
    for m in range(num_fram):
        # generate the codeword
        c = np.random.randint(0, 1, N)
        # transform the bit into 1 or -1
        c_channel = np.zeros(N)
        for k in range(N):
            if c[k] == 0:
                c_channel[k] = 1
            else:
                c_channel[k] = -1
        # noise z
        z = noise_std * randn(N)
        # add the noise to the codeword
        y = c_channel + z
        # caculate the llr for given channel output
        y_llr = np.zeros(N)
        for q in range(N):
            y_llr[q] = (y[q] * 2) / (noise_std * noise_std)
        # num of iteration
        it = 8 
        # list for save the message for v and c respectlly
        v_c = []
        c_v = []
        # initial the message for v and c
        v_c.append(SPA_function.H_V_C_initial_f(H, E, y_llr))
        c_v.append(SPA_function.H_C_V_f(H, v_c[0], E))
        # save the error for each frame
        errors = 0
        # decoding process for each iter
        for o in range(1, it):

            s = SPA_function.H_V_C_f(H, c_v[o-1], E, y_llr, False)
            for m in range(N):
                if s[m] > 0:
                    s[m] = 0
                else:
                    s[m] = 1
            # test the decode codeword is satisfied the C*H.T=0
            if (s == c).all() or o == it - 1:
                v_c.append(s)
                error = (s != c).sum()
                errors += error
                m = errors
                break
            # continue the itr if not reached the max itr
            else:
                v_c.append(SPA_function.H_V_C_f(H, c_v[o-1], E, y_llr))
                c_v.append(SPA_function.H_C_V_f(H, v_c[o], E))
        # save the num of bit error
        bit_error += m
    sum_bit = N * num_fram 
    ber = bit_error / sum_bit
    BER.append(ber)
    print(bit_error)

print(BER)    
