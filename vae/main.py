""" The main pipeline for VAEs. """

import argparse
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
sys.path.append("../")
from vae import VAE
from utils import logz
from utils import utils


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Hyperparameters that we should adjust as needed.
    p.add_argument('--batch_size', type=int, default=64, 
            help='Batch size for training VAEs')
    p.add_argument('--latent_dim', type=int, default=50,
            help='Size of the latent (i.e., z) state')
    p.add_argument('--lrate', type=float, default=0.001,
            help='Adam learning rate')
    p.add_argument('--seed', type=int, default=0,
            help='Random seed, affects numpy, random, and tf')

    # Iterations, etc.
    p.add_argument('--data_name', type=str, default='mnist',
            help='For now this is going to be mnist (lowercase)')
    p.add_argument('--log_every_t_iter', type=int, default=100,
            help='Controls the amount of time information is logged')
    p.add_argument('--snapshot_every_t_iter', type=int, default=1000,
            help='Save the model every t iterations so we can inspect later')
    p.add_argument('--test_cols', type=int, default=20,
            help='When testing MNIST, how many columns in our image grid')
    p.add_argument('--test_rows', type=int, default=20,
            help='When testing MNIST, how many rows in our image grid')
    p.add_argument('--train_iters', type=int, default=40001,
            help='Number of iterations (not the same as epoch)')
    args = p.parse_args()
    print("\nUsing the following arguments: {}".format(args))

    # Set up the directory to log things.
    log_dir = "logs/"+args.data_name+"/seed_"+str(args.seed)
    print("log_dir: {}\n".format(log_dir))
    assert not os.path.exists(log_dir), "Error: log_dir already exists!"
    logz.configure_output_dir(log_dir)
    os.makedirs(log_dir+'/snapshots/') # NN weights
    os.makedirs(log_dir+'/visuals/')   # VAE visuals
    with open(log_dir+'/args.pkl','wb') as f:
        pickle.dump(args, f)

    session = utils.get_tf_session()
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Use train/valid for training the generative model, and compare the
    # generated stuff with the held-out test data.
    data = utils.load_dataset(args.data_name)
    vae = VAE(session, data, log_dir, args)
    vae.train()
