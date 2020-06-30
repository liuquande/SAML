from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
from tensorflow.image import resize_images
# try:
#     import special_grads
# except KeyError as e:
#     print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e, file=sys.stderr)

from tensorflow.python.platform import flags
from layer import conv_block, deconv_block, fc, max_pool, concat2d
from utils import xent, kd, _get_segmentation_cost, _get_compactness_cost

class SAML:
    def __init__(self, args):
        """ Call construct_model_*() after initializing MASF"""
        self.args = args

        self.batch_size = args.meta_batch_size
        self.test_batch_size = args.test_batch_size
        self.volume_size = args.volume_size
        self.n_class = args.n_class
        self.compactness_loss_weight = args.compactness_loss_weight
        self.smoothness_loss_weight = args.smoothness_loss_weight
        self.margin = args.margin

        self.forward = self.forward_unet
        self.construct_weights = self.construct_unet_weights
        self.seg_loss = _get_segmentation_cost
        self.get_compactness_cost = _get_compactness_cost

    def construct_model_train(self, prefix='metatrain_'):
        # a: meta-train for inner update, b: meta-test for meta loss
        self.inputa = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.labela = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.n_class])
        self.inputa1= tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.labela1= tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.n_class])
        self.inputb = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.labelb = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.n_class])
        self.input_group = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.label_group = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.n_class])
        self.contour_group = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], 1])
        self.metric_label_group = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.training_mode = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving")


        self.clip_value = self.args.gradients_clip_value
        self.KEEP_PROB = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            def task_metalearn(inp, reuse=True):
                # Function to perform meta learning update """
                inputa, inputa1, inputb, labela, labela1, labelb, input_group, contour_group, metric_label_group = inp

                # Obtaining the conventional task loss on meta-train
                task_outputa, _, _ = self.forward(inputa, weights, is_training=self.training_mode)
                task_lossa = self.seg_loss(task_outputa, labela)
                task_outputa1, _, _ = self.forward(inputa1, weights, is_training=self.training_mode)
                task_lossa1 = self.seg_loss(task_outputa1, labela1)

                ## perform inner update with plain gradient descent on meta-train
                grads = tf.gradients((task_lossa + task_lossa1)/2.0, list(weights.values()))
                grads = [tf.stop_gradient(grad) for grad in grads] # first-order gradients approximation
                gradients = dict(zip(weights.keys(), grads))
                # fast_weights = dict(zip(weights.keys(), [weights[key] - self.inner_lr * gradients[key] for key in weights.keys()]))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.inner_lr * tf.clip_by_norm(gradients[key], clip_norm=self.clip_value) for key in weights.keys()]))

                ## compute compactness loss
                task_outputb, task_predmaskb, _ = self.forward(inputb, fast_weights, is_training=self.training_mode)
                task_lossb = self.seg_loss(task_outputb, labelb)
                compactness_loss_b, length, area, boundary_b = self.get_compactness_cost(task_outputb, labelb)
                compactness_loss_b = self.compactness_loss_weight * compactness_loss_b

                # compute smoothness loss
                _, _, embeddings = self.forward(input_group, fast_weights, is_training=self.training_mode)
                coutour_embeddings = self.extract_coutour_embedding(contour_group, embeddings)
                metric_embeddings = self.forward_metric_net(coutour_embeddings)

                print (metric_label_group.shape)
                print (metric_embeddings.shape)
                smoothness_loss_b = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=metric_label_group[..., 0], embeddings=metric_embeddings, margin=self.margin)
                smoothness_loss_b = self.smoothness_loss_weight * smoothness_loss_b
                task_output = [task_lossb, compactness_loss_b, smoothness_loss_b, task_predmaskb, boundary_b, length, area, task_lossa, task_lossa1]

                return task_output

            self.global_step = tf.Variable(0, trainable=False)
            # self.inner_lr = tf.train.exponential_decay(learning_rate=self.args.inner_lr, global_step=self.global_step, decay_steps=self.args.decay_step, decay_rate=self.args.decay_rate)
            # self.outer_lr = tf.train.exponential_decay(learning_rate=self.args.outer_lr, global_step=self.global_step, decay_steps=self.args.decay_step, decay_rate=self.args.decay_rate)
            self.inner_lr = tf.Variable(self.args.inner_lr, trainable=False)
            self.outer_lr = tf.Variable(self.args.outer_lr, trainable=False)
            self.metric_lr = tf.Variable(self.args.metric_lr, trainable=False)

            input_tensors = (self.inputa, self.inputa1, self.inputb, self.labela, self.labela1, self.labelb, self.input_group, self.contour_group, self.metric_label_group)
            result = task_metalearn(inp=input_tensors)
            self.seg_loss_b, self.compactness_loss_b, self.smoothness_loss_b, self.task_predmaskb, self.boundary_b, self.length, self.area, self.seg_loss_a, self.seg_loss_a1= result
           
        ## Performance & Optimization
        if 'train' in prefix:
            self.source_loss = (self.seg_loss_a + self.seg_loss_a1) / 2.0
            self.target_loss = self.seg_loss_b + self.compactness_loss_b + self.smoothness_loss_b

            var_list_segmentor = [v for v in tf.trainable_variables() if 'metric' not in v.name.split('/')]
            var_list_metric = [v for v in tf.trainable_variables() if 'metric' in v.name.split('/')]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.inner_lr).minimize(self.source_loss, global_step=self.global_step)

            optimizer = tf.train.AdamOptimizer(self.outer_lr)
            gvs = optimizer.compute_gradients(self.target_loss, var_list=var_list_segmentor)

            # observe stability of gradients for meta loss
            # l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            # for grad, var in gvs:
            #     tf.summary.histogram("gradients_norm/" + var.name, l2_norm(grad))
            #     tf.summary.histogram("feature_extractor_var_norm/" + var.name, l2_norm(var))
            #     tf.summary.histogram('gradients/' + var.name, var)
            #     tf.summary.histogram("feature_extractor_var/" + var.name, var)

            # gvs = [(grad, var) for grad, var in gvs]
            gvs = [(tf.clip_by_norm(grad, clip_norm=self.clip_value), var) for grad, var in gvs]
            self.meta_train_op = optimizer.apply_gradients(gvs)

            # for grad, var in gvs:
            #     tf.summary.histogram("gradients_norm_clipped/" + var.name, l2_norm(grad))
            #     tf.summary.histogram('gradients_clipped/' + var.name, var)

            self.metric_train_op = tf.train.AdamOptimizer(self.metric_lr).minimize(self.smoothness_loss_b, var_list=var_list_metric)

        ## Summaries
        # scalar_summaries = []
        # train_images = []
        # val_images = []

        tf.summary.scalar(prefix+'source_1 loss', self.seg_loss_a)
        tf.summary.scalar(prefix+'source_2 loss', self.seg_loss_a1)
        tf.summary.scalar(prefix+'target_loss', self.seg_loss_b)
        tf.summary.scalar(prefix+'target_coutour_loss', self.compactness_loss_b)
        tf.summary.scalar(prefix+'target_length', self.length)
        tf.summary.scalar(prefix+'target_area', self.area)
        tf.summary.image("meta_test_mask", tf.expand_dims(tf.cast(self.task_predmaskb, tf.float32), 3))
        tf.summary.image("meta_test_gth", tf.expand_dims(tf.cast(self.labelb[:,:,:,1], tf.float32), 3))
        tf.summary.image("meta_test_image", tf.expand_dims(tf.cast(self.inputb[:,:,:,1], tf.float32), 3))
        tf.summary.image("meta_test_boundary", tf.expand_dims(tf.cast(self.boundary_b[:,:,:], tf.float32), 3))
        tf.summary.image("meta_test_ct_bg_sample", tf.expand_dims(tf.cast(self.contour_group[:,:,:, 0], tf.float32), 3))
        tf.summary.image("meta_input_group", tf.expand_dims(tf.cast(self.input_group[:,:,:, 1], tf.float32), 3))
        tf.summary.image("label_group", tf.expand_dims(tf.cast(self.label_group[:,:,:, 1], tf.float32), 3))

    def extract_coutour_embedding(self, coutour, embeddings):

        coutour_embeddings = coutour * embeddings
        average_embeddings = tf.reduce_sum(coutour_embeddings, [1,2])/tf.reduce_sum(coutour, [1,2])
        # print (coutour.shape)
        # print (embeddings.shape)
        # print (coutour_embeddings.shape)
        # print (average_embeddings.shape)
        return average_embeddings

    def construct_model_test(self, prefix='test'):
        self.test_input = tf.placeholder("float", shape=[self.test_batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.test_label = tf.placeholder("float", shape=[self.test_batch_size, self.volume_size[0], self.volume_size[1], self.n_class])

        with tf.variable_scope('model', reuse=None) as testing_scope:
            if 'weights' in dir(self):
                testing_scope.reuse_variables()
                weights = self.weights
            else:
                raise ValueError('Weights not initilized. Create training model before testing model')

            outputs, mask, _ = self.forward(self.test_input, weights)
            losses = self.seg_loss(outputs, self.test_label)
            # self.pred_prob = tf.nn.softmax(outputs)
            self.outputs = mask

        self.test_loss = losses
        # self.test_acc = accuracies

    def forward_metric_net(self, x):

        with tf.variable_scope('metric', reuse=tf.AUTO_REUSE) as scope:

            w1 = tf.get_variable('w1', shape=[48,24])
            b1 = tf.get_variable('b1', shape=[24])
            out = fc(x, w1, b1, activation='leaky_relu')
            w2 = tf.get_variable('w2', shape=[24,16])
            b2 = tf.get_variable('b2', shape=[16])
            out = fc(out, w2, b2, activation='leaky_relu')

        return out

    def construct_unet_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)

        with tf.variable_scope('conv1') as scope:
            weights['conv11_weights'] = tf.get_variable('weights', shape=[5, 5, 3, 16], initializer=conv_initializer)
            weights['conv11_biases'] = tf.get_variable('biases', [16])
            weights['conv12_weights'] = tf.get_variable('weights2', shape=[5, 5, 16, 16], initializer=conv_initializer)
            weights['conv12_biases'] = tf.get_variable('biases2', [16])

        with tf.variable_scope('conv2') as scope:
            weights['conv21_weights'] = tf.get_variable('weights', shape=[5, 5, 16, 32], initializer=conv_initializer)
            weights['conv21_biases'] = tf.get_variable('biases', [32])
            weights['conv22_weights'] = tf.get_variable('weights2', shape=[5, 5, 32, 32], initializer=conv_initializer)
            weights['conv22_biases'] = tf.get_variable('biases2', [32])
        ## Network has downsample here

        with tf.variable_scope('conv3') as scope:
            weights['conv31_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)
            weights['conv31_biases'] = tf.get_variable('biases', [64])
            weights['conv32_weights'] = tf.get_variable('weights2', shape=[3, 3, 64, 64], initializer=conv_initializer)
            weights['conv32_biases'] = tf.get_variable('biases2', [64])

        with tf.variable_scope('conv4') as scope:
            weights['conv41_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)
            weights['conv41_biases'] = tf.get_variable('biases', [128])
            weights['conv42_weights'] = tf.get_variable('weights2', shape=[3, 3, 128, 128], initializer=conv_initializer)
            weights['conv42_biases'] = tf.get_variable('biases2', [128])
        ## Network has downsample here

        with tf.variable_scope('conv5') as scope:
            weights['conv51_weights'] = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=conv_initializer)
            weights['conv51_biases'] = tf.get_variable('biases', [256])
            weights['conv52_weights'] = tf.get_variable('weights2', shape=[3, 3, 256, 256], initializer=conv_initializer)
            weights['conv52_biases'] = tf.get_variable('biases2', [256])

        with tf.variable_scope('deconv6') as scope:
            weights['deconv6_weights'] = tf.get_variable('weights0', shape=[3, 3, 128, 256], initializer=conv_initializer)
            weights['deconv6_biases'] = tf.get_variable('biases0', shape=[128], initializer=conv_initializer)
            weights['conv61_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 128], initializer=conv_initializer)
            weights['conv61_biases'] = tf.get_variable('biases', [128])
            weights['conv62_weights'] = tf.get_variable('weights2', shape=[3, 3, 128, 128], initializer=conv_initializer)
            weights['conv62_biases'] = tf.get_variable('biases2', [128])

        with tf.variable_scope('deconv7') as scope:
            weights['deconv7_weights'] = tf.get_variable('weights0', shape=[3, 3, 64, 128], initializer=conv_initializer)
            weights['deconv7_biases'] = tf.get_variable('biases0', shape=[64], initializer=conv_initializer)
            weights['conv71_weights'] = tf.get_variable('weights', shape=[3, 3, 128, 64], initializer=conv_initializer)
            weights['conv71_biases'] = tf.get_variable('biases', [64])
            weights['conv72_weights'] = tf.get_variable('weights2', shape=[3, 3, 64, 64], initializer=conv_initializer)
            weights['conv72_biases'] = tf.get_variable('biases2', [64])

        with tf.variable_scope('deconv8') as scope:
            weights['deconv8_weights'] = tf.get_variable('weights0', shape=[3, 3, 32, 64], initializer=conv_initializer)
            weights['deconv8_biases'] = tf.get_variable('biases0', shape=[32], initializer=conv_initializer)
            weights['conv81_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 32], initializer=conv_initializer)
            weights['conv81_biases'] = tf.get_variable('biases', [32])
            weights['conv82_weights'] = tf.get_variable('weights2', shape=[3, 3, 32, 32], initializer=conv_initializer)
            weights['conv82_biases'] = tf.get_variable('biases2', [32])

        with tf.variable_scope('deconv9') as scope:
            weights['deconv9_weights'] = tf.get_variable('weights0', shape=[3, 3, 16, 32], initializer=conv_initializer)
            weights['deconv9_biases'] = tf.get_variable('biases0', shape=[16], initializer=conv_initializer)
            weights['conv91_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 16], initializer=conv_initializer)
            weights['conv91_biases'] = tf.get_variable('biases', [16])
            weights['conv92_weights'] = tf.get_variable('weights2', shape=[3, 3, 16, 16], initializer=conv_initializer)
            weights['conv92_biases'] = tf.get_variable('biases2', [16])

        with tf.variable_scope('output') as scope:
            weights['output_weights'] = tf.get_variable('weights', shape=[3, 3, 16, 2], initializer=conv_initializer)
            weights['output_biases'] = tf.get_variable('biases', [2])

        return weights

    def forward_unet(self, inp, weights, is_training=True):

        self.conv11 = conv_block(inp, weights['conv11_weights'], weights['conv11_biases'], scope='conv1/bn1', bn=False, is_training=is_training)
        self.conv12 = conv_block(self.conv11, weights['conv12_weights'], weights['conv12_biases'], scope='conv1/bn2', is_training=is_training)
        self.pool11 = max_pool(self.conv12, 2, 2, 2, 2, padding='VALID')
        # 192x192x16
        self.conv21 = conv_block(self.pool11, weights['conv21_weights'], weights['conv21_biases'], scope='conv2/bn1', is_training=is_training)
        self.conv22 = conv_block(self.conv21, weights['conv22_weights'], weights['conv22_biases'], scope='conv2/bn2', is_training=is_training)
        self.pool21 = max_pool(self.conv22, 2, 2, 2, 2, padding='VALID')
        # 96x96x32
        self.conv31 = conv_block(self.pool21, weights['conv31_weights'], weights['conv31_biases'], scope='conv3/bn1', is_training=is_training)
        self.conv32 = conv_block(self.conv31, weights['conv32_weights'], weights['conv32_biases'], scope='conv3/bn2', is_training=is_training)
        self.pool31 = max_pool(self.conv32, 2, 2, 2, 2, padding='VALID')
        # 48x48x64
        self.conv41 = conv_block(self.pool31, weights['conv41_weights'], weights['conv41_biases'], scope='conv4/bn1', is_training=is_training)
        self.conv42 = conv_block(self.conv41, weights['conv42_weights'], weights['conv42_biases'], scope='conv4/bn2', is_training=is_training)
        self.pool41 = max_pool(self.conv42, 2, 2, 2, 2, padding='VALID')
        # 24x24x128
        self.conv51 = conv_block(self.pool41, weights['conv51_weights'], weights['conv51_biases'], scope='conv5/bn1', is_training=is_training)
        self.conv52 = conv_block(self.conv51, weights['conv52_weights'], weights['conv52_biases'], scope='conv5/bn2', is_training=is_training)
        # 24x24x256

        ## add upsampling, meanwhile, channel number is reduced to half
        self.deconv6 = deconv_block(self.conv52, weights['deconv6_weights'], weights['deconv6_biases'], scope='deconv/bn6', is_training=is_training)
        # 48x48x128
        self.sum6 = concat2d(self.deconv6, self.deconv6)
        self.conv61 = conv_block(self.sum6, weights['conv61_weights'], weights['conv61_biases'], scope='conv6/bn1', is_training=is_training)
        self.conv62 = conv_block(self.conv61, weights['conv62_weights'], weights['conv62_biases'], scope='conv6/bn2', is_training=is_training)
        # 48x48x128

        self.deconv7 = deconv_block(self.conv62, weights['deconv7_weights'], weights['deconv7_biases'], scope='deconv/bn7', is_training=is_training)
        # 96x96x64
        self.sum7 = concat2d(self.deconv7, self.deconv7)
        self.conv71 = conv_block(self.sum7, weights['conv71_weights'], weights['conv71_biases'], scope='conv7/bn1', is_training=is_training)
        self.conv72 = conv_block(self.conv71, weights['conv72_weights'], weights['conv72_biases'], scope='conv7/bn2', is_training=is_training)
        # 96x96x64

        self.deconv8 = deconv_block(self.conv72, weights['deconv8_weights'], weights['deconv8_biases'], scope='deconv/bn8', is_training=is_training)
        # 192x192x32
        self.sum8 = concat2d(self.deconv8, self.deconv8)
        self.conv81 = conv_block(self.sum8, weights['conv81_weights'], weights['conv81_biases'], scope='conv8/bn1', is_training=is_training)
        self.conv82 = conv_block(self.conv81, weights['conv82_weights'], weights['conv82_biases'], scope='conv8/bn2', is_training=is_training)
        self.conv82_resize = tf.image.resize_images(self.conv82, [384, 384], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        # 192x192x32

        self.deconv9 = deconv_block(self.conv82, weights['deconv9_weights'], weights['deconv9_biases'], scope='deconv/bn9', is_training=is_training)
        # 384x384x16
        self.sum9 = concat2d(self.deconv9, self.deconv9)
        self.conv91 = conv_block(self.sum9, weights['conv91_weights'], weights['conv91_biases'], scope='conv9/bn1', is_training=is_training)
        self.conv92 = conv_block(self.conv91, weights['conv92_weights'], weights['conv92_biases'], scope='conv9/bn2', is_training=is_training)
        # 384x384x16

        self.logits = conv_block(self.conv92, weights['output_weights'], weights['output_biases'], scope='outpu/bn', bn=False, is_training=is_training)
        #384x384x2

        self.pred_prob = tf.nn.softmax(self.logits) # shape [batch, w, h, num_classes]
        self.pred_compact = tf.argmax(self.pred_prob, axis=-1) # shape [batch, w, h]

        self.embeddings = concat2d(self.conv82_resize, self.conv92)

        return self.pred_prob, self.pred_compact, self.embeddings
