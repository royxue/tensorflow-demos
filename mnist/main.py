import input_data
import tensorflow as tf

def main():
    # Load MNIST dataset
    mnist = input_data.read_data_sets('minist_data/', one_hot=True)

    # Create TensorFlow Session
    sess = tf.InteractiveSession()

    # Placeholder for data, input and output, None will be defined later by the batch size
    x = tf.placeholder('float', shape=[None, 784])
    y_ = tf.placeholder('float', shape=[None, 10])

    # Define variables for the regression
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Intialize the viriables, then they can be used by session
    sess.run(tf.initialize_all_variables())

    # Write regression function
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    # Write cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # Train Steps
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Train data with batches
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # Validation results
    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

    # Finally output
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})



if __name__ == "__main__":
    main()
