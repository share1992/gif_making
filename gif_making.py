import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter
import os

def gradient_color_change_magic(x, y, z=None):
    if z is None:

        # fit spline
        tck, u = interpolate.splprep([x, y], k=1, s=0.0)
        x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # Gradient color change magic
        x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)

    elif z is not None:
        # fit spline
        tck, u = interpolate.splprep([x, y, z], k=1, s=0.0)
        x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # Gradient color change magic
        x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)

    return lc


def plot_irc_for_gif(path, gif_points_to_circle="all", imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled. To stitch them into a gif, use ImageMagick:
    convert -delay 8 $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param path: path to IRC energies (txt or csv file)
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    # Reading in data as Pandas DataFrame, assuming no header and delimiting whitespace
    data = pd.read_csv(path, header=None, delim_whitespace=True)

    # Converting a.u. to kcal/mol
    data[2] = (data[1] - data[1][0]) * 627.51

    # x axis is first column of data, y is third column (just created)
    x = data[0]
    y = data[2]

    if gif_points_to_circle == "all":
        gif_points_to_circle = range(len(x))

    for i in gif_points_to_circle:
        # Put all plot aesthetic details in this loop
        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, s=100, c=x, cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("Intrinsic Reaction Coordinate (Bohr$\sqrt{amu}$)", fontsize=16)
        ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=16)

        # Always set xlim and ylim. It is important that x and y limits are always going to be the same for every
        # iteration of this loop.
        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 3 * max(y), 3 * max(y))
        plt.tick_params(labelsize=14)

        # Circle the point of interest
        ax.scatter(x[i], y[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=4)
        plt.savefig(os.path.join(output_directory, imgname + "_%05d.png" % i), format='png', dpi=100)

        plt.close(fig)

    plt.close('all')


def plot_pcs_for_gif_2D(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]

    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)


        ax.scatter(x, y, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.tick_params(labelsize=14)
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        plt.savefig(os.path.join(output_directory, imgname + "_2PCs_%04d.pdf" % i), bbox_inches='tight',
                    format='pdf')

        plt.close(fig)

    plt.close('all')


def plot_pcs_for_gif_3D(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]
    z = data[2]

    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=25., azim=55)

        lc = gradient_color_change_magic(x, y, z)
        ax.add_collection(lc)
        ax.scatter(x, y, z, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16, labelpad=10)
        ax.set_ylabel("PC2", fontsize=16, labelpad=10)
        ax.set_zlabel("PC3", fontsize=16, labelpad=15)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.set_zlim(min(z) - 0.1 * max(z), 1.1 * max(z))
        ax.tick_params(labelsize=14, axis='z', pad=10)
        ax.tick_params(labelsize=14, axis='x')
        ax.tick_params(labelsize=14, axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], z[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        # Set viewing "distance" to plot, so axes aren't cut off in final images
        ax.dist = 9
        plt.savefig(os.path.join(output_directory, imgname + "_3PCs_%04d.pdf" % i), format='pdf')

        plt.close(fig)

    plt.close('all')


def plot_pcs_for_gif_3D_rotating(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]
    z = data[2]

    rot = 0
    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))
        rot += 360/len(x)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=25., azim=rot)

        lc = gradient_color_change_magic(x, y, z)
        ax.add_collection(lc)
        ax.scatter(x, y, z, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16, labelpad=10)
        ax.set_ylabel("PC2", fontsize=16, labelpad=10)
        ax.set_zlabel("PC3", fontsize=16, labelpad=20)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.set_zlim(min(z) - 0.1 * max(z), 1.1 * max(z))
        ax.tick_params(labelsize=14, axis='z', pad=10)
        ax.tick_params(labelsize=14, axis='x')
        ax.tick_params(labelsize=14, axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], z[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        # Set viewing "distance" to plot, so axes aren't cut off in final images
        ax.dist = 11
        plt.savefig(os.path.join(output_directory, imgname + "_3PCs_rotating_%04d.pdf" % i), format='pdf')

        plt.close(fig)

    plt.close('all')