# from save_model.pvt_CAM_channel_att_upscale import build_model
import os 
import tensorflow as tf 
# from metrics.metrics_last import  iou_metric, MAE, WFbetaMetric, SMeasure, Emeasure,  dice_coef, iou_metric
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
from tensorflow.keras.utils import get_custom_objects
from model_research import build_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_dataset(route):
    X_path = '{}/images/'.format(route)
    Y_path = '{}/masks/'.format(route)
    X_full = sorted(os.listdir(f'{route}/images'))
    Y_full = sorted(os.listdir(f'{route}/masks'))

    X_train = [X_path + x for x in X_full]
    Y_train = [Y_path + x for x in Y_full]

    test_decoder = build_decoder(with_labels=False, target_size=(img_size, img_size), ext='jpg', 
                                segment=True, ext2='jpg')
    test_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                                augmentAdv=False, augment=False, augmentAdvSeg=False)
    return test_dataset, len(X_train)

def benchmark(route, model, BATCH_SIZE = 32, save_file_name = "benchmark_result.txt"):
    
    list_of_datasets = os.listdir(route)
    f = open(save_file_name,"a")
    f.write("\n")
    for datasets in list_of_datasets:
        print(datasets, ":")
        test_dataset, len_data = load_dataset(os.path.join(route,datasets))
        steps_per_epoch = len_data // BATCH_SIZE
        loss, dice_coeff, bce_dice_loss, IoU, zero_IoU, mae = model.evaluate(test_dataset, steps=steps_per_epoch)
        f.write("{}:".format(datasets))
        f.write("dice_coeff: {}, bce_didce_loss: {}, IoU: {}, zero_IoU: {}, mae: {}".format(dice_coeff, bce_dice_loss, IoU, zero_IoU, mae))
        f.write('\n')

if __name__ == "__main__":

    img_size = 256
    BATCH_SIZE = 1
    SEED = 1024
    save_path = "best_model.h5"
    route_data = "./TestDataset/"
    path_to_test_dataset = "./TestDataset/"
    model = build_model(img_size)
    model.load_weights(save_path)

    model.compile(metrics=[dice_coeff, bce_dice_loss, IoU, zero_IoU, tf.keras.metrics.MeanSquaredError()])
    
    benchmark(path_to_test_dataset, model)
