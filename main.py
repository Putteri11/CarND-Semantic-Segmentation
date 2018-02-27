import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, w3, w4, w7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # upsampling: 2, 2 and 8

    # 1 by 1 convolution
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # upsample by 2
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # skip conneciton
    output = tf.add(output, vgg_layer4_out[:,:,:,0:num_classes])

    # upsample by 2
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # skip connection
    output = tf.add(output, vgg_layer3_out[:,:,:,0:num_classes])

    # upsample by 8
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = nn_last_layer, labels = correct_label))

    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    loss = cross_entropy_loss + regularization_loss

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #logits = nn_last_layer

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate):#, logits, num_classes):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    """
    prediction = tf.argmax(logits, 1)
    correct = tf.argmax(tf.reshape(correct_label, (-1, num_classes)), 1)

    mean_iou, _ = tf.metrics.mean_iou(correct, prediction, num_classes)
    """
    print("Training...")
    print()
    for epoch in range(epochs):
        print("Epoch {} ...".format(epoch + 1))
        t1 = time.time()
        for img, label in get_batches_fn(batch_size):
            _, _loss = sess.run([train_op, loss], feed_dict = {input_image: img, correct_label: label, keep_prob: 0.5})

            #m_iou = sess.run(mean_iou, feed_dict = {input_image: img, correct_label: label, keep_prob: 1.0})

        t2 = time.time()
        print("... took {} minutes, {} seconds".format(round((t2 - t1) / 60), round((t2 - t1) % 60)))
        print("Loss: {}".format(_loss))
        #print("Mean iou: {}".format(m_iou))
        print()

    pass
tests.test_train_nn(train_nn)

"""
def evaluate(sess):

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    
    iou = 0

    return iou
"""

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        model_file = os.path.join(data_dir, "model", str(time.time()))
        builder = tf.saved_model.builder.SavedModelBuilder(model_file)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        label = tf.placeholder(tf.float32, (None, None, None, num_classes))
        #keep_prob = tf.placeholder(tf.float32)
        rate = 0.001
        n_epochs = 2
        batch_size = 80

        input_image, keep, w3, w4, w7 = load_vgg(sess, vgg_path)

        output_layer = layers(w3, w4, w7, num_classes)

        logits, train_op, loss = optimize(nn_last_layer = output_layer, correct_label = label, learning_rate = rate, num_classes = num_classes)

        # TODO: Train NN using the train_nn function

        sess.run(tf.global_variables_initializer())

        inp = input("Train model [(Y)/n]? ")

        if (inp == "n"):
            tf.saved_model.loader.load(sess, ["fcn"], "./model")
        else:
            train_nn(sess, n_epochs, batch_size, get_batches_fn, train_op, loss, input_image, label, keep, rate)
            builder.add_meta_graph_and_variables(sess, ["fcn"])
            builder.save()


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep, input_image)

        

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
