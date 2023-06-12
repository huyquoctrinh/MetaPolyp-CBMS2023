import tensorflow as tf
import os 

# def auto_select_accelerator():
#     try:
#         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#         tf.config.experimental_connect_to_cluster(tpu)
#         tf.tpu.experimental.initialize_tpu_system(tpu)
#         strategy = tf.distribute.experimental.TPUStrategy(tpu)
#         print("Running on TPU:", tpu.master())
#     except ValueError:
#         strategy = tf.distribute.get_strategy()
#     print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
#     return strategy

def default_augment_seg(input_image, input_mask):

    input_image = tf.image.random_brightness(input_image, 0.1)
    input_image = tf.image.random_contrast(input_image, 0.9, 1.1)
    input_image = tf.image.random_saturation(input_image, 0.9, 1.1)
    input_image = tf.image.random_hue(input_image, 0.01)

    # flipping random horizontal or vertical
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    
    return input_image, input_mask

def BatchAdvAugmentSeg(imagesT, masksT):
    
    images, masks = default_augment_seg(imagesT, masksT)
    
    return images, masks

def build_decoder(with_labels=True, target_size=(256, 256), ext='png', segment=False, ext2='png'):
    
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3, dct_method='INTEGER_ACCURATE')
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3, dct_method='INTEGER_ACCURATE')
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.resize(img, target_size)
        # img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    def decode_mask(path, gray=True):
        file_bytes = tf.io.read_file(path)
        if ext2 == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext2 in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
    
        img = tf.image.rgb_to_grayscale(img) if gray else img
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    def decode_with_labels(path, label):
        return decode(path), label
    
    def decode_with_segments(path, path2, gray=True):
        return decode(path), decode_mask(path2, gray)
    
    if segment:
        return decode_with_segments
    
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
#         img = tf.image.rot90(img, k=tf.random.uniform([],0,4,tf.int32))
        
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.image.random_hue(img, 0.02)
        
        # img = transform_mat(img)
        
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, augmentAdv=False, augmentAdvSeg=False, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    #dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize)
    # dset = dset.map(BatchAdvAugment, num_parallel_calls=AUTO) if augmentAdv else dset
    dset = dset.map(BatchAdvAugmentSeg, num_parallel_calls=AUTO) if augmentAdvSeg else dset
    dset = dset.prefetch(AUTO)
    
    return dset
