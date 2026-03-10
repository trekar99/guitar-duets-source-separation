import os
import json
import matplotlib.pyplot as plt
import numpy as np


def makehistograms(name):
    metrics_file = '/gpu-data3/mgly/deguitht/' + name + '/temp/metrics.txt'

    # Specify the directory to save the plots
    save_dir = 'plots/' + name
    os.makedirs(save_dir, exist_ok=True)

    # Read the file
    with open(metrics_file, 'r') as file:
        data = file.readlines()

    all_sdr_values = []
    for line in data:
        item = json.loads(line)
        guitar1_metrics = item['metrics']['guitar1']
        guitar2_metrics = item['metrics']['guitar2']

        for song in guitar1_metrics.values():
            all_sdr_values.extend(song['SDR'])

        for song in guitar2_metrics.values():
            all_sdr_values.extend(song['SDR'])

    # Calculate minimum and maximum SDR values
    min_sdr = int(min(all_sdr_values))
    max_sdr = int(max(all_sdr_values))


    # Parse the dictionaries
    for line in data:
        item = json.loads(line)
        epoch = item['epoch']
        guitar1_metrics = item['metrics']['guitar1']
        guitar2_metrics = item['metrics']['guitar2']

        guitar1_sdr = []
        for song in guitar1_metrics.values():
            guitar1_sdr.extend(song['SDR'])

        guitar2_sdr = []
        for song in guitar2_metrics.values():
            guitar2_sdr.extend(song['SDR'])

        # Create histograms
        bin_edges = np.arange(min_sdr, max_sdr + 2, 1)   # Customize the bin range as needed

        fig, ax = plt.subplots()
        ax.hist(guitar1_sdr, bins=bin_edges, alpha=0.5, label='Guitar1')
        ax.hist(guitar2_sdr, bins=bin_edges, alpha=0.5, label='Guitar2')
        ax.set_xlabel('SDR')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram for Epoch {epoch}')
        ax.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'hist_epoch_{epoch}.png'))
        plt.close()

    print(f"Plots saved to directory: {save_dir}")