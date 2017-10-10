import tensorflow as tf
import main.operations as myops
from tensorflow.python.ops import math_ops as tfmath

import numpy as np

array = tf.placeholder(tf.int32, shape=[6,])
# init = tf.global_variables_initializer()
mean = myops.my_reduce_mean(array)

A = tf.constant([[2.0, -1.0],[-1.0, 2.0]])
b = tf.constant([3.0, 0.0], shape=[2,1])
x = tf.Variable(tf.zeros([2,1]), tf.float32) # Optimizing variable

quadratic_term = tfmath.multiply(1 / 2, tfmath.matmul(tfmath.matmul(tf.transpose(x), A), x))
linear_term = tfmath.matmul(tf.transpose(b), x)
cost = tfmath.add(quadratic_term, linear_term)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

grads_and_vars = optimizer.compute_gradients(cost)
grad_norms = [tf.nn.l2_loss(g) for g, v in grads_and_vars]
grad_norm = tf.add_n(grad_norms)
opt_operation = optimizer.apply_gradients(grads_and_vars)

init_op = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init_op)
    for i in range(10000):
        _, c, norm = session.run([opt_operation, cost, grad_norm])
        # if norm < 0.0000001:
        #     print("Got it")
        #     break
        if i % 1000 == 0:
            print (x.eval())
            print (c)
            print (norm)

