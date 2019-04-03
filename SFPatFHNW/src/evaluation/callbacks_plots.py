import numpy as np
import os
import tensorflow as tf

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: Several methods that calculate the dir and then ensure it exists is not great... extract!


def residualsplot_on_epoch_end(original_callback, epoch, y_true, y_pred):
    dir =f'{original_callback.dir}/plots'
    if not os.path.isdir(dir):
        os.makedirs(dir)

    y_diff = y_true - y_pred

    # plot and save file
    plt.style.use("ggplot")
    fig = plt.figure()
    plt.xlim(1e-9, 1e-3)
    plt.ylim(-1e-2, 1e-2)
    plt.plot(y_pred, y_diff, '.')
    plt.title(f"Predicted Value vs Residuals (xlim(1e-9, 1e-3) & (-1e-2, 1e-2))")
    plt.xlabel("predictions")
    plt.ylabel("residuals")
    plt.savefig(f'{dir}/res_vs_pred_limited_{epoch}.png')
    plt.close(fig)

    # plot and save file, unlimited
    plt.style.use("ggplot")
    fig = plt.figure()
    plt.plot(y_pred, y_diff, '.')
    plt.title(f"Predicted Value vs Residuals (Unlimited)")
    plt.xlabel("predictions")
    plt.ylabel("residuals")
    plt.savefig(f'{dir}/res_vs_pred_unlimited_{epoch}.png')
    plt.close(fig)


def boxplots_on_epoch_end(original_callback, epoch, y_true, y_pred):
    dir =f'{original_callback.dir}/plots'
    if not os.path.isdir(dir):
        os.makedirs(dir)

    # plot and save file
    plt.style.use("ggplot")
    fig = plt.figure()
    plt.title(f"Box Plot y max 1e-3")
    plt.ylim(top=1e-3)
    plt.boxplot([y_pred, y_true], labels=["pred", "true"])
    plt.savefig(f'{dir}/boxplot_limited_{epoch}.png')
    plt.close(fig)

    plt.style.use("ggplot")
    fig = plt.figure()
    plt.title(f"Box Plot")
    plt.boxplot([y_pred, y_true], labels=["pred", "true"])
    plt.savefig(f'{dir}/boxplot_unlimited_{epoch}.png')
    plt.close(fig)
