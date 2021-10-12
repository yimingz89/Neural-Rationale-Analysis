#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# font config
rcParams['font.family'] = "Fira Sans"
rcParams['font.sans-serif'] = ["Fira Sans", "DejaVu Sans"]
rcParams['font.weight'] = "regular"

def plot_heatmap(count, scores=None, column_labels=None, row_labels=None,
                 output_path="plot.png"):
    """
    Plotting function that can be used to visualize (self-)attention
    Args:
        scores: attention scores
        column_labels: labels for columns (e.g. target tokens)
        row_labels: labels for rows (e.g. source tokens)
        output_path: path to save to

    Returns:

    """
    assert output_path.endswith(".png") or output_path.endswith(".pdf"), \
        "output path must have .png or .pdf extension"

    col_sent_len = len(column_labels) if column_labels is not None else 1
    row_sent_len = len(row_labels) if row_labels is not None else 1
    scores = scores[:row_sent_len, :col_sent_len]
    #save_path = 'results/sst/bernoulli_sparsity01505/test_perturbed_scores/score' + str(count) + '.npy'
    #np.save(save_path, scores.numpy()[0])

    # automatic label size
    labelsize = 25 * (10 / max(col_sent_len, row_sent_len))

    matplotlib.rcParams['xtick.labelsize'] = labelsize
    matplotlib.rcParams['ytick.labelsize'] = labelsize

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    heatmap = plt.imshow(scores, cmap='viridis', aspect='equal',
                         origin='upper', vmin=0., vmax=1.)

    # display values
    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            # if scores[row, col] == 0. or scores[row, col] == 1. or True:
            plt.text(
                col, row, '%.2f' % scores[row, col],
                horizontalalignment='center', verticalalignment='center',
                color='w', fontsize=6)

    if column_labels is not None:
        ax.set_xticklabels(column_labels, minor=False, rotation="vertical")

    if row_labels is not None:
        ax.set_yticklabels(row_labels, minor=False)

    ax.xaxis.tick_top()

    if column_labels is not None:
        ax.set_xticks(np.arange(scores.shape[1]) + 0, minor=False)
    else:
        plt.xticks([], [])

    if row_labels is not None:
        ax.set_yticks(np.arange(scores.shape[0]) + 0, minor=False)
    else:
        plt.yticks([], [])

    plt.tight_layout()

    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(fig)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path = output_path + ".png"
        plt.savefig(output_path)

    plt.close()
