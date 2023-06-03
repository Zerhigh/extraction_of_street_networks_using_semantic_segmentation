import matplotlib.pyplot as plt
import glob
import json


filepath = 'D:/SHollendonner/segmentation_results/'
models = ['2605_512_unet_densenet201_MS_200epochs_full_continued'] #['unet3plus_0704_256_basicunet', 'unet3plus_0904_256_basicunet_MS', 'unet3plus_0804_256_attunet']

def plot_metrics(model):
    """
        receives: model name
        return: True
        extracts and plots the metrics of a trained model
    """

    fig1, ax1 = plt.subplots(1, len(models), figsize=(len(models) * 10, 10), sharex=True, sharey=True)
    metrics = glob.glob(f'{filepath}/{model}/checkpoints/*.txt')
    print(metrics)

    with open(metrics[0], 'r') as f:
        for line in f:
            # string slicing necessary to remove " " from beginning and end of json.dump
            json_acceptable_string = line[1:-1].replace("'", "\"")
            metric = json.loads(json_acceptable_string)
            if len(models) > 1:
                ax1[i].plot(range(len(metric['val_hybrid_loss'])), metric['val_hybrid_loss'], label='val_hybrid_loss')
                ax1[i].plot(range(len(metric['hybrid_loss'])), metric['hybrid_loss'], label='hybrid_loss')
                ax1[i].plot(range(len(metric['val_iou_seg'])), metric['val_iou_seg'], label='val_iou_seg')
                ax1[i].plot(range(len(metric['iou_seg'])), metric['iou_seg'], label='iou_seg')
                # ax1[i].plot(range(len(metric['val_loss'])), metric['val_loss'], label='val_loss')
                # ax1[i].plot(range(len(metric['loss'])), metric['loss'], label='loss')
                ax1[i].set_title(model)
                ax1[i].legend()
            else:
                ax1.plot(range(len(metric['val_hybrid_loss'])), metric['val_hybrid_loss'], label='val_hybrid_loss')
                ax1.plot(range(len(metric['hybrid_loss'])), metric['hybrid_loss'], label='hybrid_loss')
                ax1.plot(range(len(metric['val_iou_seg'])), metric['val_iou_seg'], label='val_iou_seg')
                ax1.plot(range(len(metric['iou_seg'])), metric['iou_seg'], label='iou_seg')
                # ax1.plot(range(len(metric['val_loss'])), metric['val_loss'], label='val_loss')
                # ax1.plot(range(len(metric['loss'])), metric['loss'], label='loss')
                ax1.set_title(model)
                ax1.legend()

    plt.savefig(f'{filepath}/{model}/metrics_plot.png')

    return True

for i, model in enumerate(models):
    plot_metrics(model)

plt.show()