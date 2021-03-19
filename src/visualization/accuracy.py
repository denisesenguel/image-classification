import matplotlib.pyplot as plt

def accuracy(history):

    """
    Produces training and validation accuracy plots from estimated image classification models

    Args:
        history: A tensorflow.python.keras.callbacks.History object (fitted model)
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, acc, label='Training Accuracy')
    plt.plot(history.epoch, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, loss, label='Training Loss')
    plt.plot(history.epoch, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()