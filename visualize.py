import os 
import tensorflow as tf 
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
from tensorflow.keras.utils import get_custom_objects
from model import build_model
import cv2 
import numpy as np
# from save_model.best_up_ca import build_model
# from save_model.ca_best_msf import build_model
import matplotlib.pyplot as plt 
os.environ["CUDA_VISIBLE_DEVICES"]="2"



def load_dataset(route, img_size = 256):
    BATCH_SIZE = 1
    X_path = '{}/images/'.format(route)
    Y_path = '{}/masks/'.format(route)
    X_full = sorted(os.listdir(f'{route}/images'))
    Y_full = sorted(os.listdir(f'{route}/masks'))

    X_train = [X_path + x for x in X_full]
    Y_train = [Y_path + x for x in Y_full]

    test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    test_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                                augmentAdv=False, augment=False, augmentAdvSeg=False, shuffle = None)
    return test_dataset, len(X_train)

def predict(model, dataset, len_data, outdir ="./save_vis/Etis/"):
    steps_per_epoch = len_data//1
    masks = model.predict(dataset, steps=steps_per_epoch)
    # print(masks.shape)
    i = 0
    for x, y in dataset:
        print(y[0].shape)
        # print(i, masks[i].shape)
        a = masks[i]
        mask_new = np.dstack([a, a, a])
        # print(x.shape, y.shape)
        gt = np.dstack([y[0], y[0], y[0]])
        # gt = cv2.cvtColor(y[0], cv2.COLOR_GRAY2RGB)
        # true = cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB)
        im_h = np.concatenate([x[0], gt * 255, mask_new *255], axis = 1)
        cv2.imwrite("{}/{}.jpg".format(outdir, i), im_h)
        i+=1

def visualize(src_dir, model, outdir ="./save_vis/Etis/"):
    dataset, len_data = load_dataset(src_dir)
    predict(model, dataset, len_data, outdir)

if __name__ == "__main__":
    
    BATCH_SIZE = 16
    img_size = 256
    SEED = 1024
    save_path = "best_model.h5"
    route_data = "./TestDataset/"
    outdir ="./save_vis/cvc300/"
    src_dir = "./TestDataset/CVC-300"

    model = build_model(img_size)
    model.load_weights(save_path)

    visualize(src_dir, model, outdir)
    