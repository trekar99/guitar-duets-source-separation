import re
import matplotlib.pyplot as plt


def create_sdr(name):
    file_name = '/home/mgly/guitarduetstrans/'+name+'.txt'
    # Read the text file
    with open(file_name, 'r') as file:
        lines = file.readlines()

        epochs = []
        guitar1_median_sdr = []
        guitar1_median_sisdr = []
        guitar1_median_sir = []
        guitar1_median_isr = []
        guitar1_median_sar = []
        guitar2_median_sdr = []
        guitar2_median_sisdr = []
        guitar2_median_sir = [] 
        guitar2_median_isr = [] 
        guitar2_median_sar = []


    for line in lines:
        if 'LOSS train' in line:
            values = line.split(" ")
            epoch = values[3].replace(',','')
        if 'Median SDR for guitar1' in line:
            epochs.append(int(epoch))
            guitar1_median_sdr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SI-SDR for guitar1:' in line:
            guitar1_median_sisdr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SIR for guitar1:' in line:
            guitar1_median_sir.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median ISR for guitar1:' in line:
            guitar1_median_isr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SAR for guitar1:' in line:
            guitar1_median_sar.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SDR for guitar2:' in line:
            guitar2_median_sdr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SI-SDR for guitar2:' in line:
            guitar2_median_sisdr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SIR for guitar2:' in line:
            guitar2_median_sir.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median ISR for guitar2:' in line:
            guitar2_median_isr.append(float(line.split(" ")[-1].replace('\n','')))
        if 'Median SAR for guitar2:' in line:
            guitar2_median_sar.append(float(line.split(" ")[-1].replace('\n','')))

    print(epochs)
    print(guitar1_median_sdr)
    #print(guitar1_median_sisdr)
    #print(guitar1_median_sir)
    #print(guitar1_median_isr)
    #print(guitar1_median_sar)
    print(guitar2_median_sdr)
    #print(guitar2_median_sisdr)
    #print(guitar2_median_sir)
    #print(guitar2_median_isr) 
    #print(guitar2_median_sar)


    # Plot guitar1 metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, guitar1_median_sdr, label='SDR')
    plt.plot(epochs, guitar1_median_sisdr, label='SI-SDR')
    plt.plot(epochs, guitar1_median_sir, label='SIR')
    plt.plot(epochs, guitar1_median_isr, label='ISR')
    plt.plot(epochs, guitar1_median_sar, label='SAR')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Metrics for Guitar1')
    plt.legend()
    plt.grid()
    plt.savefig('plots/'+name+'/guitar1_metrics.png')
    plt.close()

    # Plot guitar2 metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, guitar2_median_sdr, label='SDR')
    plt.plot(epochs, guitar2_median_sisdr, label='SI-SDR')
    plt.plot(epochs, guitar2_median_sir, label='SIR')
    plt.plot(epochs, guitar2_median_isr, label='ISR')
    plt.plot(epochs, guitar2_median_sar, label='SAR')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Metrics for Guitar2')
    plt.legend()
    plt.grid()
    plt.savefig('plots/'+name+'/guitar2_metrics.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, guitar1_median_sdr, label='Guitar1 SDR')
    plt.plot(epochs, guitar2_median_sdr, label='Guitar2 SDR')
    plt.xlabel('Epochs')
    plt.ylabel('SDR')
    plt.title('SDR Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('plots/'+name+'/sdr_comparison.png')
    plt.close()