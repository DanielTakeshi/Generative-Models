"""
Variational Autoencoder.
"""

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


class VAE:

    def __init__(self, session, data, log_dir, args):
        self.sess = session
        self.bsize = args.batch_size
        self.data_mb_list = utils.list_of_minibatches(data, self.bsize)
        self.log_dir = log_dir
        self.args = args

        # Manage shapes and placeholders. E.g. for MNIST, O = odim = 784.
        self.odim = data['X_train'].shape[1]
        assert len(data['X_train'].shape) == 2
        self.data_BO = tf.placeholder(tf.float32, [None,self.odim])

        # Reparam. trick, have Gaussian noise as a separate "input".
        self.std_norm_BZ = tf.placeholder(tf.float32, [None,args.latent_dim])
        self.e_mean_BZ, self.e_logstd_BZ = \
                self._encoder(self.data_BO, args.latent_dim)

        # In test-time application, we can explicitly assign to this instead of
        # feeding something to the two previous placeholders in the graph.
        self.latent_BZ = self.e_mean_BZ + (tf.exp(self.e_logstd_BZ) * self.std_norm_BZ)
        self.d_mean_BO, self.d_logstd_BO = \
                self._decoder(self.latent_BZ, o_dim=self.odim)

        # Form the training objective for VAEs. 
        self.kldiv = self._compute_KL()
        self.log_p = self._compute_logprob()
        self.neg_lb_llhd = - (self.log_p - self.kldiv)
        self.train_op = tf.train.AdamOptimizer(args.lrate).minimize(self.neg_lb_llhd)

        # Collect the weights, view summary, and initialize.
        self.weights     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.weights_v   = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.w_shapes    = [w.get_shape().as_list() for w in self.weights]
        self.num_weights = np.sum([np.prod(sh) for sh in self.w_shapes])
        self.enc_weights = [v for v in self.weights if 'Encoder' in v.name]
        self.dec_weights = [v for v in self.weights if 'Decoder' in v.name]
        self._print_summary()
        self.sess.run(tf.global_variables_initializer())


    def _compute_KL(self):
        """ 
        For KL divergence, each element in the minibatch (MB) produces a mean
        and log std vector (so do `exp` and then square to get diagonal of the
        covariance). Take average over the MB to get the estimate of kldiv.
        Finally, don't forget that the determinant of a diagonal matrix (which
        we have here) is the product of the diagonals. Draw this out by hand if
        you're confused. :-) 
        
        I use three tf.reduce_sums which should probably be put into one
        command, but this is only for ease of exposition.
        """
        k = self.args.latent_dim
        self.traces_B = tf.reduce_sum(tf.square(tf.exp(self.e_logstd_BZ)), axis=1)
        self.e_mean_B = tf.reduce_sum(tf.square(self.e_mean_BZ), axis=1)
        self.sums_B   = tf.reduce_sum(self.e_logstd_BZ, axis=1)
        self.kldivs_B = 0.5 * (self.traces_B + self.e_mean_B - k - 2*self.sums_B)
        return tf.reduce_mean(self.kldivs_B)


    def _compute_logprob(self):
        """ 
        We compute the logprobs for each component in the MB, and _then_ return
        the mean for the "official" quantity. Each MB element x' has prob based
        on mean x and logstd vector from the decoder.

        log p(x) = - 0.5 * log((2*pi)^k * det(Sigma))
                   - 0.5 * (x-mu)^T * Sigma^{-1} * (x-mu)

        Be careful, e.g. the logstd has \sigma_i terms, not their square, but
        the log cancels that out with the 0.5 factor anyway.
        """
        k = self.args.latent_dim
        self.sq_diff_BO = tf.square(self.d_mean_BO - self.data_BO)
        self.var_BO     = tf.square(tf.exp(self.d_logstd_BO))
        self.first_B    = - (k/2) * tf.log(2*np.pi) - tf.reduce_sum(self.d_logstd_BO, axis=1)
        self.second_B   = - tf.reduce_sum(self.sq_diff_BO / (2*self.var_BO), axis=1)
        self.logprobs_B = self.first_B + self.second_B # No tf.log needed here
        return tf.reduce_mean(self.logprobs_B)


    def _encoder(self, e_input_BO, z_dim):
        """ Builds the Encoder network. For now use FC nets.
        
        Takes as input our original data, and outputs latent features to capture
        meaningful factors of variation in data. Well, in theory. In reality, it
        needs to output mean and (diagonal) covariance vectors, to make the
        process differentiable. I will write an extensive blog post about this.
        """
        with tf.variable_scope('Encoder', reuse=False):
            self.e_Bh1 = tf.nn.relu(tf.layers.dense(e_input_BO, 512))
            self.e_Bh2 = tf.nn.relu(tf.layers.dense(self.e_Bh1, 256))
            e_mean_BZ   = tf.layers.dense(self.e_Bh2, z_dim)
            e_logstd_BZ = tf.layers.dense(self.e_Bh2, z_dim)
            return (e_mean_BZ, e_logstd_BZ)


    def _decoder(self, z_input_BZ, o_dim):
        """ Builds the Decoder network. For now use FC nets.
        
        Takes as input the `z` variable or the latent features, and outputs
        (ideally) what we want, such as images. Thus, it's sampling something
        conditional on the z. Oh and also we still want the mean and log std...
        """
        with tf.variable_scope('Decoder', reuse=False):
            self.d_Bh1 = tf.nn.relu(tf.layers.dense(z_input_BZ, 256))
            self.d_Bh2 = tf.nn.relu(tf.layers.dense(self.d_Bh1, 512))
            d_mean_BO   = tf.layers.dense(self.d_Bh2, o_dim)
            d_logstd_BO = tf.layers.dense(self.d_Bh2, o_dim)
            return (d_mean_BO, d_logstd_BO)


    def train(self):
        """ Runs training. See the in-line comments. """
        args = self.args
        t_start = time.time()
        num_mbs = len(self.data_mb_list['X_train'])

        for ii in range(args.train_iters):
            # Sample minibatch + standard Gaussian noise and form feed.
            real_BO = self.data_mb_list['X_train'][ii % num_mbs]
            std_norm_BZ = np.random.standard_normal((self.bsize,args.latent_dim))
            feed = {self.data_BO: real_BO, self.std_norm_BZ: std_norm_BZ}
            _, neg_lb_loss, kldiv, log_p, first, second, logstd_BO = self.sess.run(
                    [self.train_op, self.neg_lb_llhd, self.kldiv, self.log_p,
                        self.first_B, self.second_B, self.d_logstd_BO], 
                    feed
            )

            if (ii % args.log_every_t_iter == 0):
                print("\n  ************ Iteration %i ************" % ii)
                #print("first {}".format(first))
                #print("second {}".format(second))
                elapsed_time_hours = (time.time() - t_start) / (60.0 ** 2)
                logz.log_tabular("LogProb",    log_p)
                logz.log_tabular("KlDiv",      kldiv)
                logz.log_tabular("NegLbLhd",   neg_lb_loss)
                logz.log_tabular("TimeHours",  elapsed_time_hours)
                logz.log_tabular("Iterations", ii)
                logz.dump_tabular()

            if (ii % args.snapshot_every_t_iter == 0) and (self.log_dir is not None):
                # --------------------------------------------------------------
                # See if we're making cool images and also save weights.
                # Unfortunately some of this is highly specific to MNIST...
                # Don't worry about the reshaping order because all that the
                # computer sees is just the 784-dimensional vector (for now).
                # We use a different batch size here, `bs`.
                # --------------------------------------------------------------
                bs = args.test_cols * args.test_rows
                dims = int(np.sqrt(self.odim))
                latent_BZ = np.random.standard_normal((bs,args.latent_dim))
                feed = {self.latent_BZ: latent_BZ}
                dec_out_BO, dec_logstd_BO = \
                        self.sess.run([self.d_mean_BO, self.d_logstd_BO], feed)

                # With the mean and (log) std, we can sample.
                eps_BO = np.random.standard_normal(size=dec_out_BO.shape)
                sampled_BO = dec_out_BO + (np.exp(dec_logstd_BO) * eps_BO)
                dec_out_BDD = np.reshape(sampled_BO, (bs,dims,dims))
                weights_v = self.sess.run(self.weights_v)
                self._save_snapshot(ii, weights_v, dec_out_BDD)


    #####################
    # DEBUGGING METHODS #
    #####################

    def _save_snapshot(self, ii, weights_v, out_BDD):
        """ Save the NN weights and GAN-generated images. 
        
        Careful, the image stuff assumes a lot of MNIST-specific stuff.
        """
        itr = str(ii).zfill(len(str(abs(self.args.train_iters))))
        np.save(self.log_dir+'/snapshots/weights_'+itr, weights_v)

        # The output may not be in (0,1).
        out_BDD = np.minimum(1.0, out_BDD)
        out_BDD = np.maximum(0.0, out_BDD)
        out_BDD *= 255.0

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
                        PIL.Image.fromarray(out_BDD[index,:,:]),
                        ((col+1)*ws+col*D, (row+1)*ws+row*D)
                )
                index += 1
        new_im.save(self.log_dir+'/visuals/dec_'+itr+'.png')


    def _print_summary(self):
        print("\n=== START OF SUMMARY ===\n")
        print("Total number of weights: {}.".format(self.num_weights))

        print("\nEncoder:")
        for v in self.enc_weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        print("\nDecoder:")
        for v in self.dec_weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        print("\nOutput from the networks:")
        print("e_mean_BZ:   {}".format(self.e_mean_BZ))
        print("e_logstd_BZ: {}".format(self.e_logstd_BZ))
        print("d_mean_BO:   {}".format(self.d_mean_BO))
        print("d_logstd_BO: {}".format(self.d_logstd_BO))
        print("latent_BZ:   {}".format(self.latent_BZ))

        print("\nlen(data_mb_list['X_train']): {}".format(
            len(self.data_mb_list['X_train'])))
        print("data_mb_list['X_train'][0].shape: {}".format(
            self.data_mb_list['X_train'][0].shape))
        print("\n=== DONE WITH SUMMARY ===\n")
