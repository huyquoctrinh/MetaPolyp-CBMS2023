import cv2
from model import build_model
import numpy as np 

def load_model(model_path):
    model = build_model()
    model.load_weights(model_path)
    return model

def predict_single(model, imgPath):
    img = cv2.imread(imgPath)
    img /= 255
    result = model.predict(img)
    return result

if __name__ == "__main__":
    save_path = "best_model.h5"
    img_in = "test.png"
    img_out = "out.png"

    model = load_model(save_path)
    
    mask = predict_single(model, img_out)
    mask_out = np.dstack([mask, mask, mask])
    cv2.imwrite(img_out, mask_out)