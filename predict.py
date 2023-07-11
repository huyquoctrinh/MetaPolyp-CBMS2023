import cv2
from model import build_model
import numpy as np 

def load_model(model_path):
    model = build_model()
    model.load_weights(model_path)
    return model

def predict_single(model, imgPath):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis = 0)
    # img = img / 255.
    result = model.predict(img)
    return result

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask*255

if __name__ == "__main__":
    save_path = "best_model.h5"
    img_in = "cju0qoxqj9q6s0835b43399p4.jpg"
    img_out = "out1.jpg"

    model = load_model(save_path)
    masks = predict_single(model, img_in)
    out = mask_parse(masks)
    cv2.imwrite(img_out, out)
