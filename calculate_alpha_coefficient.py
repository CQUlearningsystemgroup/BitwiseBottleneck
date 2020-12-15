# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV,MultiTaskLassoCV
import struct
import os
import math
import heapq
import sys

# # # read data
names = ['block_layer1_0_0.txt', 'block_layer1_0_1.txt', 'block_layer1_0_2.txt', 'block_layer1_1_0.txt',
         'block_layer1_1_1.txt', 'block_layer1_1_2.txt', 'block_layer1_2_0.txt', 'block_layer1_2_1.txt',
         'block_layer1_2_2.txt', 'block_layer2_0_0.txt', 'block_layer2_0_1.txt', 'block_layer2_0_2.txt',
         'block_layer2_1_0.txt', 'block_layer2_1_1.txt', 'block_layer2_1_2.txt', 'block_layer2_2_0.txt',
         'block_layer2_2_1.txt', 'block_layer2_2_2.txt', 'block_layer2_3_0.txt', 'block_layer2_3_1.txt',
         'block_layer2_3_2.txt', 'block_layer3_0_0.txt', 'block_layer3_0_1.txt', 'block_layer3_0_2.txt',
         'block_layer3_1_0.txt', 'block_layer3_1_1.txt', 'block_layer3_1_2.txt', 'block_layer3_2_0.txt',
         'block_layer3_2_1.txt', 'block_layer3_2_2.txt', 'block_layer3_3_0.txt', 'block_layer3_3_1.txt',
         'block_layer3_3_2.txt', 'block_layer3_4_0.txt', 'block_layer3_4_1.txt', 'block_layer3_4_2.txt',
         'block_layer3_5_0.txt', 'block_layer3_5_1.txt', 'block_layer3_5_2.txt', 'block_layer4_0_0.txt',
         'block_layer4_0_1.txt', 'block_layer4_0_2.txt', 'block_layer4_1_0.txt', 'block_layer4_1_1.txt',
         'block_layer4_1_2.txt', 'block_layer4_2_0.txt', 'block_layer4_2_1.txt', 'block_layer4_2_2.txt'
         ]

# setting hyperparameter
layer_num = 48
initial_quantization_bit = 5
quantization_maximum = 2.**initial_quantization_bit - 1
threshold_of_PSNR_loss = 16.
minimum_PSNR = 55. - 16.

alpha_array = np.zeros([layer_num, 8], dtype=np.float)
for t in range(0, layer_num):
    content = np.loadtxt('.../BIB_ImageNet/activations/5_bit_modified_DoReFa/'+'%s'%names[t])
    data = np.array(content, dtype=np.float).reshape(-1, 1)
    data_abs = np.abs(data)
    max_array = heapq.nlargest(5, data_abs)
    # filter out the noise
    if max_array[0] - max_array[1] > 10:
        max = max_array[1]
    else:
        max = np.max(data_abs)
    data_abs[data_abs > quantization_maximum] = quantization_maximum
    data_round = np.round(data_abs)
    data_int = np.array(data_round, dtype=np.uint8)
    data_bit = np.unpackbits(data_int, axis=1)     # obtain the codeboke
    # # The first level of searching for PSNR
    for i in np.arange(0.001, 10, 0.1):
        # parameter of alpha is the hyperparameter of lambda in the paper
        model = Lasso(alpha=i, positive=True, max_iter=1000, random_state=0)
        model.fit(data_bit, data_abs)
        coef = np.array(model.coef_)
        # calculate the PSNR
        PSNR = 0
        alpha = np.array(model.coef_).reshape((8, 1))
        data_recov = np.dot(data_bit, alpha)
        MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
        PSNR = 10 * math.log10(quantization_maximum ** 2 / MSE)

        if PSNR <= minimum_PSNR:
            # The second level of searching for PSNR
            # if the minimul lambda satisfy the minimul PSNR, then break
            if i == 0.001:
                break
            for i_p1 in np.arange(i - 0.1, i, 0.01):
                model = Lasso(alpha=i_p1, positive=True, max_iter=1000, random_state=0)
                model.fit(data_bit, data_abs)
                coef = np.array(model.coef_)
                # calculate the PSNR
                PSNR = 0
                alpha = np.array(model.coef_).reshape((8, 1))
                data_recov = np.dot(data_bit, alpha)
                MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
                PSNR = 10 * math.log10(quantization_maximum ** 2 / MSE)

                if PSNR <= minimum_PSNR:
                    # The third level of searching for PSNR
                    for i_p2 in np.arange(i_p1 - 0.01, i_p1, 0.001):
                        model = Lasso(alpha=i_p2, positive=True, max_iter=1000, random_state=0)
                        model.fit(data_bit, data_abs)
                        coef = np.array(model.coef_)
                        # calculate the PSNR
                        PSNR = 0
                        alpha = np.array(model.coef_).reshape((8, 1))
                        data_recov = np.dot(data_bit, alpha)
                        MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
                        PSNR = 10 * math.log10(quantization_maximum ** 2 / MSE)
                        if PSNR <= minimum_PSNR:
                            break
                if PSNR <= minimum_PSNR:
                    break
        if PSNR <= minimum_PSNR:
            break

    if i == 0.001:
        i_b = i
        print('The minimul scale of lambda is large.')
    else:
        i_b = i_p2
    ###########################################################################################################
    # finish searching the optimal alpha coefficient according to the threshold of PSNR loss
    ###########################################################################################################
    # The number of bit correspond to the least PSNR.
    alpha_final = i_b - 0.001 if i_b - 0.001 > 1e-5 else 1e-3
    model = Lasso(alpha=alpha_final, positive=True, max_iter=1000, random_state=0)  # 调节alpha可以实现对拟合的程度
    model.fit(data_bit, data_abs)
    coef = np.array(model.coef_)
    print('---------------------------------------')
    print(names[t])
    print('---------------------------------------')
    print('Before tuning:')
    print('The alpha :', alpha_final)

    # calculate the PSNR
    PSNR = 0
    alpha = np.array(model.coef_).reshape((8, 1))
    data_recov = np.dot(data_bit, alpha)
    MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
    PSNR = 10 * math.log10(quantization_maximum ** 2 / MSE)
    print('The least PSNR:', PSNR)
    # Count the bit number
    bit_num_zero = 0
    for m in range(0, len(coef)):
        if coef[m] != 0.:
            bit_num_zero = bit_num_zero + 1
    print('The bit number :', bit_num_zero)
    print('---------------------------------------')
    ###########################################################################################################
    # tune the PSNR loss under the same number of bits
    ###########################################################################################################
    # The first level of tuning the PSNR loss
    i_b1_threshold = i_b-10 if i_b-10 >= 0 else 0
    for i_b1 in np.arange(i_b-0.001, i_b1_threshold, -0.1):
        # if the minimul lambda satisfy the minimul PSNR, then break
        if i_b == 0.001:
            break
        # if all coefficients of alpha become zero, then break
        if bit_num_zero == initial_quantization_bit:
            break

        model = Lasso(alpha=i_b1, positive=True, max_iter=1000, random_state=0)
        model.fit(data_bit, data_abs)
        coef = np.array(model.coef_)
        bit_num = 0
        for m_1 in range(0, len(coef)):
            if coef[m_1] != 0.:
                bit_num = bit_num + 1
        # i_b1 <= 0.1 is used to prevent the ib_1 from becoming zero.
        if bit_num > bit_num_zero or i_b1 <= 0.1:
            # The second level of tuning the PSNR loss
            i_2_threshold = i_b1 - 0.1 if i_b1-0.1 >= 0 else 0
            for i_2 in np.arange(i_b1 + 0.1, i_2_threshold, -0.01):
                model = Lasso(alpha=i_2, positive=True, max_iter=1000, random_state=0)
                model.fit(data_bit, data_abs)
                coef = np.array(model.coef_)
                bit_num = 0
                for m_2 in range(0, len(coef)):
                    if coef[m_2] != 0.:
                        bit_num = bit_num + 1

                if bit_num > bit_num_zero or i_2 <= 0.01:
                    # The third level of tuning the PSNR loss
                    i_3_threshold = i_2-0.01 if i_2-0.01 >= 0 else 0
                    for i_3 in np.arange(i_2 + 0.01, i_3_threshold, -0.001):
                        model = Lasso(alpha=i_3, positive=True, max_iter=1000, random_state=0)
                        model.fit(data_bit, data_abs)
                        coef = np.array(model.coef_)
                        bit_num = 0
                        for m_3 in range(0, len(coef)):
                            if coef[m_3] != 0.:
                                bit_num = bit_num + 1
                        if bit_num > bit_num_zero or i_3 <= 0.001 + 1e-5:   # adding 0.00001 is to prevent the noise
                            print(i_3)
                            break
                if bit_num > bit_num_zero or i_2 <= 0.01:
                    break
        if bit_num > bit_num_zero or i_b1 <= 0.1:
            break

    # Get the best alpha
    print('After tuning:')
    # when reaching the minimum PSNR at the beginning or the bit number can't be compreseed
    if i == 0.001 or bit_num_zero == initial_quantization_bit:
        best_alpha = 1e-5
        print(names[t] + ' alpha = ', best_alpha)
        print('The minimum scale of lambda is big.')
    elif i_3 <= 0.001 + 1e-5:     # the minimum search range
        best_alpha = 0.001
        print('alpha = ', best_alpha)
    else:
        best_alpha = i_3 + 0.001
        print('alpha = ', best_alpha)

    model = Lasso(alpha=best_alpha, positive=True, max_iter=1000, random_state=0)
    model.fit(data_bit, data_abs)
    alpha = np.array(model.coef_).reshape((8, 1))
    print('alpha coefficient:\n', alpha)
    alpha_array[t] = alpha.reshape((1, 8))
    data_recov = np.dot(data_bit, alpha)
    # calculate the PSNR
    MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
    PSNR = 10 * math.log10(quantization_maximum ** 2 / MSE)
    print('The finial PSNR :', PSNR)
    print('\n')
    print('\n')
np.savez("./alpha_coefficient/alpha_array.npz", alpha_array)
print('It is finished ')



