"""
Do `python plot.py logs/mnist` and it should collect all the random seeds.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import sys
from os.path import join
from pylab import subplots
plt.style.use('seaborn-darkgrid')
sns.set_context(rc={'lines.markeredgewidth': 1.0})
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)

# Some matplotlib settings.
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 17
ysize = 18
xsize = 18
lw = 2
ms = 8
error_region_alpha = 0.3

# Attributes to include in a plot.
ATTRIBUTES = ["LogProb", "KlDiv", "NegLbLhd", "TimeHours"]
COLORS = ['red', 'blue', 'yellow', 'black', 'orange']


def plot_one_directory(args, dirnames, figname):
    """ 
    Here, `dirname` contains directories named by the random seed.  Then inside
    each of those seeds, there's a `log.txt` file.
    """
    logdir = args.logdir
    num = len(ATTRIBUTES)
    fig, axes = subplots(num, figsize=(13,4*num))

    for (dd, cc) in zip(dirnames, COLORS):
        A = np.genfromtxt(join(logdir, dd, 'log.txt'), 
                          delimiter='\t', 
                          dtype=None, 
                          names=True)
        x = A['Iterations']

        for (i,attr) in enumerate(ATTRIBUTES):
            axes[i].plot(x, A[attr], '-', lw=lw, color=cc, label=dd)
            axes[i].set_ylabel(attr, fontsize=ysize)
            axes[i].tick_params(axis='x', labelsize=tick_size)
            axes[i].tick_params(axis='y', labelsize=tick_size)
            axes[i].legend(loc='best', ncol=2, prop={'size':legend_size})

        axes[0].set_ylim([-10,10])
        axes[1].set_ylim([-10,10])
        axes[2].set_ylim([0,10])
        axes[3].set_ylim([0,10])

    plt.tight_layout()
    plt.savefig(figname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help="Example, logs/mnist/")
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    assert args.logdir[-1] == '/'

    dirnames = sorted(os.listdir(args.logdir))
    dirnames = dirnames[:args.limit] # Too many seeds makes it hard to see
    figname = FIGDIR+args.logdir[:-1]+'.png' # Get rid of trailing slash.
    figname = figname.replace('logs/', '')

    print("plotting to: {}\nwith these seeds: {}".format(figname, dirnames))
    plot_one_directory(args, dirnames, figname)
