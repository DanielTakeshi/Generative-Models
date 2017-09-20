""" Basic GAN. """

import numpy as np
import PIL
import sys
import tensorflow as tf
import tensorflow.contrib as tc
import time
sys.path.append("../")
from utils import logz
from utils import utils
from collections import defaultdict
from PIL import Image
from PIL import ImageDraw
np.set_printoptions(edgeitems=100, linewidth=200, suppress=True, precision=5)


class GAN:
    """ A vanilla Generative Adversarial Network. """

    def __init__(self, session, data, log_dir, args):
        self.sess = session
        self.bsize = args.batch_size
        self.data_mb_list = utils.list_of_minibatches(data, self.bsize)
        self.log_dir = log_dir
        self.args = args

        # Manage shapes and placeholders. E.g. for MNIST, O = odim = 784.
        self.prior_dim = args.gen_prior_size
        self.odim = data['X_train'].shape[1]
        assert len(data['X_train'].shape) == 2

        # Note that the input for the Generator is a sampled Gaussian.
        self.D_data_BO = tf.placeholder(shape=[self.bsize,self.odim], dtype=tf.float32)
        self.G_data_BP = tf.placeholder(shape=[None,self.prior_dim], dtype=tf.float32)

        # Get output from the networks, re-using the Discriminator's weights.
        self.D_real_B = self._build_D(self.D_data_BO, reuse=False)
        self.G_out_BO = self._build_G(self.G_data_BP, self.odim)
        self.D_fake_B = self._build_D(self.G_out_BO, reuse=True) # Generator input!

        # Collect the weights. Naming is key to separate weight updates.
        self.weights   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.weights_v = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.w_shapes  = [w.get_shape().as_list() for w in self.weights]
        self.num_weights = np.sum([np.prod(sh) for sh in self.w_shapes])
        self.dis_weights = [v for v in self.weights if 'Discriminator' in v.name]
        self.gen_weights = [v for v in self.weights if 'Generator' in v.name]

        # Needed for the API of TF's cross entropy method.
        sh = self.D_real_B.get_shape() 
        ones = tf.ones(shape=sh, dtype=tf.float32)

        # Training operations. Note the use of variable names.
        self.loss_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_B,
                                                        labels=ones*args.one_sided_ls) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_B, labels=ones*0.)
        )
        self.loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_B, labels=ones)
        )

        self.train_D_op = tf.train.AdamOptimizer(args.lrate_d).minimize(
                self.loss_D, var_list=self.dis_weights)
        self.train_G_op = tf.train.AdamOptimizer(args.lrate_g).minimize(
                self.loss_G, var_list=self.gen_weights)

        # View a summary and initialize.
        self._print_summary()
        self.sess.run(tf.global_variables_initializer())


    def _build_D(self, d_input_BO, reuse):
        """ Builds the Discriminator network using a FC network. """
        with tf.variable_scope('Discriminator', reuse=reuse):
            self.d_Nh1 = tf.nn.relu(tf.layers.dense(d_input_BO, 1024))
            self.d_Nh2 = tf.nn.relu(tf.layers.dense(self.d_Nh1, 512))
            self.d_result_B1 = tf.layers.dense(self.d_Nh2, 1)
            self.d_result_B = tf.reshape(self.d_result_B1, [-1])
            return self.d_result_B


    def _build_G(self, g_input_BP, outdim):
        """ Builds the Generator network using a FC network. """
        with tf.variable_scope('Generator', reuse=False):
            self.g_Bh1 = tf.nn.relu(tf.layers.dense(g_input_BP, 256))
            self.g_Bh2 = tf.nn.relu(tf.layers.dense(self.g_Bh1, 512))
            self.g_result_BO = tf.layers.dense(self.g_Bh2, outdim)
            return self.g_result_BO

    
    def _sample_prior(self):
        """ Sample prior for the Generator. 
        
        This might need to be more sophisticated than just assuming IID
        everywhere ... and I should probably generate a huge block of noise
        beforehand to avoid this computation during training.
        """
        if self.args.gen_prior_type == 'gaussian':
            return np.random.standard_normal(size=(self.bsize,self.prior_dim))
        elif self.args.gen_prior_type == 'uniform':
            return np.random.uniform(-1.0, 1.0, size=(self.bsize,self.prior_dim))


    def train(self):
        """ Runs training. See the in-line comments. """
        args = self.args
        t_start = time.time()
        num_mbs = len(self.data_mb_list['X_train'])

        for ii in range(args.train_iters):
            stats = defaultdict(list)

            # Sample minibatch and form feed dictionary.
            real_BO = self.data_mb_list['X_train'][ii % num_mbs]
            prior_BP = self._sample_prior()
            feed = {self.D_data_BO: real_BO, self.G_data_BP: prior_BP}

            # Update Generator and Discriminator, I think they should be separate. 
            _, loss_D = self.sess.run([self.train_D_op, self.loss_D], feed)
            _, loss_G = self.sess.run([self.train_G_op, self.loss_G], feed)

            if (ii % args.log_every_t_iter == 0):
                print("\n  ************ Iteration %i ************" % ii)
                # --------------------------------------------------------------
                # Logging. Also record time and get a fresh set of real vs fake
                # images to evaluate (but NOT train) the Discriminator.
                # --------------------------------------------------------------
                new_feed = {self.D_data_BO: self.data_mb_list['X_train'][(ii+1) % num_mbs], 
                            self.G_data_BP: self._sample_prior()}
                dout_real, dout_fake = \
                        self.sess.run([self.D_real_B, self.D_fake_B], new_feed)
                num_correct = np.sum(dout_real > 0.0) + np.sum(dout_fake < 0.0)
                elapsed_time_hours = (time.time() - t_start) / (60.0 ** 2)

                logz.log_tabular("AvgRealScore",  np.mean(dout_real))
                logz.log_tabular("AvgFakeScore",  np.mean(dout_fake))
                logz.log_tabular("LossDis",       loss_D)
                logz.log_tabular("LossGen",       loss_G)
                logz.log_tabular("DisNumCorrect", num_correct)
                logz.log_tabular("TimeHours",     elapsed_time_hours)
                logz.log_tabular("Iterations",    ii)
                logz.dump_tabular()

            if (ii % args.snapshot_every_t_iter == 0) and (self.log_dir is not None):
                # --------------------------------------------------------------
                # See if we're making cool images and also save weights.
                # Unfortunately some of this is highly specific to MNIST...
                # Don't worry about the reshaping order because all that the
                # computer sees is just the 784-dimensional vector (for now).
                # --------------------------------------------------------------
                bs = args.test_cols * args.test_rows
                dims = int(np.sqrt(self.odim))
                prior = np.random.standard_normal(size=(bs,self.prior_dim))
                gen_out_BO = self.sess.run(self.G_out_BO, {self.G_data_BP: prior})
                gen_out_BDD = np.reshape(gen_out_BO, (bs,dims,dims))
                weights_v = self.sess.run(self.weights_v)
                self._save_snapshot(ii, weights_v, gen_out_BDD)


    #####################
    # DEBUGGING METHODS #
    #####################

    def _save_snapshot(self, ii, weights_v, gen_out_BDD):
        """ Save the NN weights and GAN-generated images. 
        
        Careful, the image stuff assumes a lot of MNIST-specific stuff.
        """
        itr = str(ii).zfill(len(str(abs(self.args.train_iters))))
        np.save(self.log_dir+'/snapshots/weights_'+itr, weights_v)

        # Generator's output may not be in (0,1).
        gen_out_BDD = np.minimum(1.0, gen_out_BDD)
        gen_out_BDD = np.maximum(0.0, gen_out_BDD)
        gen_out_BDD *= 255.0

        # Make the images human-readable using PIL.
        D = int(np.sqrt(self.odim))
        rows = self.args.test_rows
        cols = self.args.test_cols
        ws = 5 # I.e., whitespace
        im_w = cols*D + (cols+1)*ws
        im_h = rows*D + (rows+1)*ws

        # Create a blank white image and repeatedly paste stuff on it.
        new_im = Image.new(mode='L', size=(im_w,im_h), color=255)
        draw = ImageDraw.Draw(new_im)
        index = 0
        for row in range(rows):
            for col in range(cols):
                new_im.paste(
                        PIL.Image.fromarray(gen_out_BDD[index,:,:]),
                        ((col+1)*ws+col*D, (row+1)*ws+row*D)
                )
                index += 1
        new_im.save(self.log_dir+'/visuals/gen_'+itr+'.png')


    def _print_summary(self):
        print("\n=== START OF SUMMARY ===\n")
        print("Total number of weights: {}.".format(self.num_weights))

        print("\nStuff by the Discriminator's step:")
        for v in self.dis_weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        print("\nStuff updated by the Generator's step:")
        for v in self.gen_weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        print("\nOutput from the networks:")
        print("D_real_B: {}".format(self.D_real_B))
        print("D_fake_B: {}".format(self.D_fake_B))
        print("G_out_BO: {}".format(self.G_out_BO))

        print("\nlen(data_mb_list['X_train']): {}".format(
            len(self.data_mb_list['X_train'])))
        print("data_mb_list['X_train'][0].shape: {}".format(
            self.data_mb_list['X_train'][0].shape))
        print("\n=== DONE WITH SUMMARY ===\n")
