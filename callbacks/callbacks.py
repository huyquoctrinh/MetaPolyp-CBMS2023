import math
import tensorflow as tf
import matplotlib.pyplot as plt

def cosine_annealing_with_warmup(epochIdx):
    aMax, aMin = max_lr, min_lr
    warmupEpochs, stagnateEpochs, cosAnnealingEpochs = 0, 0, cos_anne_ep
    epochIdx = epochIdx % (warmupEpochs + stagnateEpochs + cosAnnealingEpochs)
    if(epochIdx < warmupEpochs):
        return aMin + (aMax - aMin) / (warmupEpochs - 1) * epochIdx
    else:
        epochIdx -= warmupEpochs
    if(epochIdx < stagnateEpochs):
        return aMax
    else:
        epochIdx -= stagnateEpochs
    return aMin + 0.5 * (aMax - aMin) * (1 + math.cos((epochIdx + 1) / (cosAnnealingEpochs + 1) * math.pi))

def plt_lr(step, schedulers):
    x = range(step)
    y = [schedulers(_) for _ in x]

    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

def get_callbacks(monitor, mode, save_path, _max_lr, _min_lr, _cos_anne_ep, save_weights_only):
    global max_lr
    max_lr = _max_lr
    global min_lr
    min_lr = _min_lr
    global cos_anne_ep
    cos_anne_ep = _cos_anne_ep

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=60,
        restore_best_weights=True,
        mode=mode
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.2,
        patience=50,
        verbose=1,
        mode=mode,
        min_lr=1e-5,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq="epoch",
    )

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=0)

    csv_logger = tf.keras.callbacks.CSVLogger('training.csv')

    callbacks = [checkpoint, csv_logger, reduce_lr]
# , reduce_lr
    return callbacks