import tensorflow as tf
import numpy as np
from absl import flags



alpha_array = np.load("./alpha_coefficient/psnr_loss_8db.npy")
initial_bit_num = 8
quantization_maximum = 2.**6
momentum = 0.1


def _save_mat(name, tensor_x):
    print(name)
    print(tensor_x.shape)
    f = open(name.decode('utf-8') + '.txt', 'w')

    for i in range(32):
        for j in range(tensor_x.shape[3]):
            v_2d = tensor_x[i, :, :, j]
            w = v_2d.shape[0]
            h = v_2d.shape[1]
            for Ii in range(w):
                for Ji in range(h):
                    strNum = str(v_2d[Ii, Ji])  
                    f.write(strNum)
                    pass
                    pass
                    f.write('\n')
    f.close()

    return tensor_x


def _tensor_print(name, tensor_x, i):
    _save_mat(name, tensor_x)
    v_2d = tensor_x[0, :, :, i]

    return v_2d


def tensor_print(name, tensor_x):
    """This function is used to print the activations, which will be save as a .txt files.

    :param name: name of activation
    :param tensor_x: activation tensor
    :return: none
    """
    for i in range(1):
        tensor_1 = tf.py_func(_tensor_print, [name, tensor_x, i], tf.float32)  # 调用
        conv1out = tf.reshape(tensor_1, shape=[1, tf.shape(tensor_1)[0], tf.shape(tensor_1)[1], 1])  # 2d->4d
        tf.summary.image(name, conv1out, max_outputs=64)
    return



def alpha_update(alpha, origin_alpha, name):
    """This function is used to update the alpha variable.

    :param alpha: alpha variable
    :param origin_alpha: the initial alpha variable
    :param name: the name of alpha variable
    :return: the updated alpha variable
    """

    origin_alpha = tf.cast(origin_alpha, tf.float32)

    alpha_update_coef = tf.get_variable(name=name, shape=None, dtype=tf.float32,
                                            initializer=tf.ones_like(alpha))
    alpha = origin_alpha*(1-momentum) + momentum*tf.multiply(alpha, alpha_update_coef)
    
    # alpha quantization
    round_back = beta
    round_infer = tf.round(beta)
    beta_round = round_back + tf.stop_gradient(round_infer - round_back)

    return alpha



def bitwise_information_bottleneck(x, name):
    """This function is the complete implement of the Bitwise Information Bottleneck quantization operation

    :param x: activation
    :param name: the name of activation
    :return: an optimized quantized activation
    """

     # load the initial alpha variable
    if name == 'block_layer1_0_0':
        init_alpha = alpha_array[0].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_0_1':
        init_alpha = alpha_array[1].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_0_2':
        init_alpha = alpha_array[2].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_1_0':
        init_alpha = alpha_array[3].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_1_1':
        init_alpha = alpha_array[4].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_1_2':
        init_alpha = alpha_array[5].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_2_0':
        init_alpha = alpha_array[6].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_2_1':
        init_alpha = alpha_array[7].reshape(initial_bit_num, 1)

    elif name == 'block_layer1_2_2':
        init_alpha = alpha_array[8].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_0_0':
        init_alpha = alpha_array[9].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_0_1':
        init_alpha = alpha_array[10].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_0_2':
        init_alpha = alpha_array[11].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_1_0':
        init_alpha = alpha_array[12].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_1_1':
        init_alpha = alpha_array[13].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_1_2':
        init_alpha = alpha_array[14].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_2_0':
        init_alpha = alpha_array[15].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_2_1':
        init_alpha = alpha_array[16].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_2_2':
        init_alpha = alpha_array[17].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_3_0':
        init_alpha = alpha_array[18].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_3_1':
        init_alpha = alpha_array[19].reshape(initial_bit_num, 1)

    elif name == 'block_layer2_3_2':
        init_alpha = alpha_array[20].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_0_0':
        init_alpha = alpha_array[21].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_0_1':
        init_alpha = alpha_array[22].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_0_2':
        init_alpha = alpha_array[23].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_1_0':
        init_alpha = alpha_array[24].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_1_1':
        init_alpha = alpha_array[25].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_1_2':
        init_alpha = alpha_array[26].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_2_0':
        init_alpha = alpha_array[27].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_2_1':
        init_alpha = alpha_array[28].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_2_2':
        init_alpha = alpha_array[29].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_3_0':
        init_alpha = alpha_array[30].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_3_1':
        init_alpha = alpha_array[31].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_3_2':
        init_alpha = alpha_array[32].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_4_0':
        init_alpha = alpha_array[33].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_4_1':
        init_alpha = alpha_array[34].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_4_2':
        init_alpha = alpha_array[35].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_5_0':
        init_alpha = alpha_array[36].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_5_1':
        init_alpha = alpha_array[37].reshape(initial_bit_num, 1)

    elif name == 'block_layer3_5_2':
        init_alpha = alpha_array[38].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_0_0':
        init_alpha = alpha_array[39].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_0_1':
        init_alpha = alpha_array[40].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_0_2':
        init_alpha = alpha_array[41].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_1_0':
        init_alpha = alpha_array[42].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_1_1':
        init_alpha = alpha_array[43].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_1_2':
        init_alpha = alpha_array[44].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_2_0':
        init_alpha = alpha_array[45].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_2_1':
        init_alpha = alpha_array[46].reshape(initial_bit_num, 1)

    elif name == 'block_layer4_2_2':
        init_alpha = alpha_array[47].reshape(initial_bit_num, 1)

    else:
        print('There is something wrong !')

    alpha_constant = tf.constant(init_alpha)
    with tf.compat.v1.variable_scope('bit_bottle'):
        alpha = tf.Variable(init_alpha, name='bit_beta', dtype=tf.float32, trainable=True)
    alpha_backward = tf.reshape(tf.constant(np.ones(shape=(initial_bit_num, 1), dtype=np.float32)), shape=(initial_bit_num, 1))
    # update the alpha coefficient
    alpha = alpha_update(alpha, alpha_constant, name=name + '_up')

    rank = x.get_shape().ndims
    assert rank is not None

    maximum = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
    x_normal = (x/maximum) * 0.5 + tf.random_uniform(tf.shape(x), minval=-0.5 / quantization_maximum, maxval=0.5 / quantization_maximum)

    round_backward = x_normal * quantization_maximum
    round_inference = tf.round(x_normal * quantization_maximum)
    x = round_backward + tf.stop_gradient(round_inference - round_backward)

    x_sign = tf.sign(x)
    x_shape = x.shape
    x = tf.multiply(x, x_sign)
    x = tf.reshape(x, [-1])
    # obtain the codebook of initial quantization
    fdiv_back_0 = tf.div(x, 2.)
    fdiv_forward_0 = tf.floordiv(x, 2.)
    x_fdiv2 = fdiv_back_0 + tf.stop_gradient(fdiv_forward_0 - fdiv_back_0)
    xbit0 = x + tf.stop_gradient(tf.subtract(x, tf.multiply(x_fdiv2, 2.)) - x)

    fdiv_back_1 = tf.div(x_fdiv2, 2.)
    fdiv_forward_1 = tf.floordiv(x_fdiv2, 2.)
    x_fdiv4 = fdiv_back_1 + tf.stop_gradient(fdiv_forward_1 - fdiv_back_1)
    xbit1 = x + tf.stop_gradient(tf.subtract(x_fdiv2, tf.multiply(x_fdiv4, 2.)) - x)

    fdiv_back_2 = tf.div(x_fdiv4, 2.)
    fdiv_forward_2 = tf.floordiv(x_fdiv4, 2.)
    x_fdiv8 = fdiv_back_2 + tf.stop_gradient(fdiv_forward_2 - fdiv_back_2)
    xbit2 = x + tf.stop_gradient(tf.subtract(x_fdiv4, tf.multiply(x_fdiv8, 2.)) - x)

    fdiv_back_3 = tf.div(x_fdiv8, 2.)
    fdiv_forward_3 = tf.floordiv(x_fdiv8, 2.)
    x_fdiv16 = fdiv_back_3 + tf.stop_gradient(fdiv_forward_3 - fdiv_back_3)
    xbit3 = x + tf.stop_gradient(tf.subtract(x_fdiv8, tf.multiply(x_fdiv16, 2.)) - x)

    fdiv_back_4 = tf.div(x_fdiv16, 2.)
    fdiv_forward_4 = tf.floordiv(x_fdiv16, 2.)
    x_fdiv32 = fdiv_back_4 + tf.stop_gradient(fdiv_forward_4 - fdiv_back_4)
    xbit4 = x + tf.stop_gradient(tf.subtract(x_fdiv16, tf.multiply(x_fdiv32, 2.)) - x)

    fdiv_back_5 = tf.div(x_fdiv32, 2.)
    fdiv_forward_5 = tf.floordiv(x_fdiv32, 2.)
    x_fdiv64 = fdiv_back_5 + tf.stop_gradient(fdiv_forward_5 - fdiv_back_5)
    xbit5 = x + tf.stop_gradient(tf.subtract(x_fdiv32, tf.multiply(x_fdiv64, 2.)) - x)

    fdiv_back_6 = tf.div(x_fdiv64, 2.)
    fdiv_forward_6 = tf.floordiv(x_fdiv64, 2.)
    x_fdiv128 = fdiv_back_6 + tf.stop_gradient(fdiv_forward_6 - fdiv_back_6)
    xbit6 = x + tf.stop_gradient(tf.subtract(x_fdiv64, tf.multiply(x_fdiv128, 2.)) - x)

    fdiv_back_7 = tf.div(x_fdiv128, 2.)
    fdiv_forward_7 = tf.floordiv(x_fdiv128, 2.)
    x_fdiv256 = fdiv_back_7 + tf.stop_gradient(fdiv_forward_7 - fdiv_back_7)
    xbit7 = x + tf.stop_gradient(tf.subtract(x_fdiv128, tf.multiply(x_fdiv256, 2.)) - x)

    xbit_stack = tf.stack([xbit7, xbit6, xbit5, xbit4, xbit3, xbit2, xbit1, xbit0], axis=1)
    # restore the optimized quantized activation
    x_optimized = tf.matmul(xbit_stack, alpha)

    x_optimized_backward = tf.div(tf.matmul(xbit_stack, alpha_backward), tf.cast(initial_bit_num, tf.float32))
    x_output = x_optimized_backward + tf.stop_gradient(x_optimized - x_optimized_backward)

    x_output = tf.reshape(x_output, shape=[-1, x_shape[-3], x_shape[-2], x_shape[-1]])
    x_output = tf.multiply(x_output, x_sign)

    output = x_output / quantization_maximum + 0.5
    output = tf.clip_by_value(output, 0.0, 1.0)
    output = output - 0.5
    output = 2 * maximum * output


    return output


def dorefa_quantization(x, name):
    rank = x.get_shape().ndims
    assert rank is not None

    maximum = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
    x_normal = (x/maximum) * 0.5 + tf.random_uniform(tf.shape(x), minval=-0.5 / quantization_maximum, maxval=0.5 / quantization_maximum)

    round_backward = x_normal
    round_inference = tf.round(x_normal * quantization_maximum) / quantization_maximum
    x_round = round_backward + tf.stop_gradient(round_inference - round_backward)

    # print the activations
    # x_print = x_normal * quantization_maximum
    # tensor_print(x_print, name)

    output = x_round + 0.5
    output = tf.clip_by_value(output, 0.0, 1.0)
    output = output - 0.5

    output = output * maximum * 2

    return output


def iterative_rounding(x):

    x_shape = x.shape
    x_sign = tf.sign(x)
    x = tf.multiply(x, x_sign)

    maximum = tf.reduce_max(tf.reshape(x, [-1]), axis=0)
    round_backward = tf.div(x, maximum) * quantization_maximum
    round_inference = tf.round(tf.div(x, maximum) * quantization_maximum)
    x_round = round_backward + tf.stop_gradient(round_inference - round_backward)

    x = tf.reshape(x_round, shape=[-1, x_shape[-3], x_shape[-2], x_shape[-1]])
    output = tf.multiply(x, x_sign)

    return output