# To run this script using Python3.6 (enmcomp3,4,11),
# assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 myPlot.py
#   source deactivate tf-py3.6

# Author: M.W. Mak, Dept. of EIE, HKPolyU
# Last update: Oct. 2017

import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from sklearn.manifold import TSNE


# Here is a utility function used to display the transformed dataset.
# The color of each point refers to the actual digit (of course,
# this information was not used by the dimensionality reduction algorithm).
# For general classification problem (not MNIST digit recognition), colors
# contain the class labels
def scatter2D(x, colors):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    labels = np.unique(colors)
    for i in labels:
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def scatter3D(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a 3D scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = Axes3D(f)
    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)
    ax.axis('tight')

    return f, ax, sc


# Display 25 images in a 5x5 grid
def show_imgs(imgs):
    cnt = 0
    r, c = 5, 5
    fi, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            ax[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            ax[i, j].axis('off')
            cnt += 1
    plt.show()


# Plot histograms of the the two outputs of a binary classifier
def plot_hist(x, nbins=50):

    fig = plt.figure()
    ax = plt.subplot(111)

    # the histogram of the data
    h1, bins = np.histogram(x[:, 0], bins=nbins, normed=1)
    ax.plot(bins[0:len(bins)-1], h1, 'r', linewidth=1, label='Target')
    h2, bins = np.histogram(x[:, 1], bins=nbins, normed=1)
    ax.plot(bins[0:len(bins)-1], h2, 'b', linewidth=1, label='Nontarget')
    ax.set_ylabel('Probability')
    
    # Shrink current axis by 20%    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


if __name__ == '__main__':
    # Test plot_hist()
    N = 100000
    mu1, sigma1 = 10, 15
    mu2, sigma2 = -10, 10
    x1 = np.asarray(mu1 + sigma1 * np.random.randn(N))
    x2 = np.asarray(mu2 + sigma2 * np.random.randn(N))
    x = np.hstack([x1.reshape((N, 1)), x2.reshape(N,1)])
    plot_hist(x, 100)

    # Test scatter2D()
    N = 100
    x = np.vstack([np.random.normal(1, 1, (N,2)),
                   np.random.normal(-1, 1, (N,2)),
                   np.random.normal(0, 0.5, (N,2))])
    y = [0] * N + [1] * N + [2] * N
    y = np.asarray(y)
    scatter2D(x, y)
    
    # Demo on t-SNE plot
    data = sio.loadmat('data/IS09_emotion/emodb_full.mat')
    X = data['x']
    y = np.squeeze(data['y'])
    X_prj = TSNE(n_components=2).fit_transform(X)
    scatter2D(X_prj, y)

    plt.show()

    
