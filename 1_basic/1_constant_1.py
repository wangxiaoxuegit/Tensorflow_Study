
import tensorflow as tf

# create graph
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# run graph in session
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
