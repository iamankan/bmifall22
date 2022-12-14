import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt

def print_history(history, savename=f'dummy'):
    print(f'Printing history, {history.keys}')
    # summarize history for accuracy
    plt.plot(history['binary_accuracy'])
    plt.plot(history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(f'{savename}_accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(f'{savename}_loss.png')

def main():
    fname = f'my_history_1670087179796380400.npy'
    history = np.load(fname, allow_pickle=True)
    mydict = history.item()
    print(type(mydict))
    for key in mydict.keys():
        print(key)
        print(mydict[key])
        break
    print_history(history=mydict, savename='1670087179796380400')
    

    

if __name__ == '__main__':
    main()