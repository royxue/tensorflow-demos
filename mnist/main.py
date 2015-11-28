import input_data
import tensorflow as tf


def main():
    # Load MNIST dataset
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

    # Create TensorFlow Session
    sess = tf.InteractiveSession()

    # Placeholder for data, input and label, None will be defined later by
    # the batch size
    x = tf.placeholder('float', shape=[None, 784])

    y_ = tf.placeholder('float', shape=[None, 10])

    # Define variables for the regression
    w = tf.Variable(tf.zeros([784, 10]))
    w_hist = tf.histogram_summary('weights', w)
    b = tf.Variable(tf.zeros([10]))
    b_hist = tf.histogram_summary('bias', b)

    # Write regression function
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    # Write cost function
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # Train Steps
    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    # Initialize TensorBoard
    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/Users/royxue/Code/TensorFlow/tensorflow-demos/mnist/logs', sess.graph_def)
    total_step = 0

    # Intialize the viriables, then they can be used by session
    sess.run(tf.initialize_all_variables())

    # Train data with batches
    for i in range(1000):
        total_step += 1
        batch_x, batch_y = mnist.train.next_batch(50)
        feed = {x: batch_x, y_: batch_y}
        sess.run(train_step, feed_dict=feed)
        if total_step % 100 == 0:
            summary_str = sess.run(merged_summary_op, feed_dict=feed)
            summary_writer.add_summary(summary_str, total_step)

    # Validation results
    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

    # Finally output
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

if __name__ == "__main__":
    main()
