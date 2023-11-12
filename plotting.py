from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    QUESTION = '2a'

    test_loss = np.load(f'output/{QUESTION}/test_loss.npy')
    train_loss = np.load(f'output/{QUESTION}/train_loss.npy')
    epochs = len(train_loss)

    plt.title(f"Question {QUESTION}")
    plt.plot(list(range(epochs)), test_loss, label="Test loss")
    plt.plot(list(range(epochs)), train_loss, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.show()
