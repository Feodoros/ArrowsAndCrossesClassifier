import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import tensorflow as tf

# Visualizing several images from training dataset.
def visualize_data(DATA_FOLDER, CATEGORY_1, CATEGORY_2):
    
    path_to_training_data_1 = os.path.join(DATA_FOLDER + '/train/' + CATEGORY_1)
    path_to_training_data_2 = os.path.join(DATA_FOLDER + '/train/' + CATEGORY_2)
    path_to_validation_data_1 = os.path.join(DATA_FOLDER + '/validation/'  + CATEGORY_1)
    path_to_validation_data_2 = os.path.join(DATA_FOLDER + '/validation/' + CATEGORY_2)
    path_to_test_data_1 = os.path.join(DATA_FOLDER + '/test/'  + CATEGORY_1)
    path_to_test_data_2 = os.path.join(DATA_FOLDER + '/test/' + CATEGORY_2)

    list_of_training_data_1 = os.listdir(path_to_training_data_1)
    list_of_training_data_2 = os.listdir(path_to_training_data_2)
    list_of_validation_data_1 = os.listdir(path_to_validation_data_1)
    list_of_validation_data_2 = os.listdir(path_to_validation_data_2)
    list_of_test_data_1 = os.listdir(path_to_test_data_1)
    list_of_test_data_2 = os.listdir(path_to_test_data_2)

    print(CATEGORY_1 + " training data length: ", len(list_of_training_data_1))
    print(CATEGORY_2 + " training data length: ", len(list_of_training_data_2))
    print(CATEGORY_1 + " validation data length: ", len(list_of_validation_data_1))
    print(CATEGORY_2 + " validation data length: ", len(list_of_validation_data_2))
    print(CATEGORY_1 + " test data length: ", len(list_of_test_data_1))
    print(CATEGORY_2 + " test data length: ", len(list_of_test_data_2))

    nrows = 4
    ncols = 4

    random.shuffle(list_of_training_data_1)
    random.shuffle(list_of_training_data_2)

    fig = plt.gcf()
    fig.set_size_inches(nrows * 2, ncols * 2)

    list_of_paths_to_training_data_1 = [os.path.join(path_to_training_data_1, fname) 
                                            for fname in list_of_training_data_1[:8]]
    list_of_paths_to_training_data_2 = [os.path.join(path_to_training_data_2, fname) 
                                            for fname in list_of_training_data_2[:8]]

    for i, path in enumerate(list_of_paths_to_training_data_1 + list_of_paths_to_training_data_2):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('Off')
        pic = img.imread(path)
        plt.imshow(pic)
    plt.show()

# Calculating accuracy, precision, recall, f1 score.
def AccRecPrec(predictions, test_labels):

    x = tf.placeholder(tf.int32, )
    y = tf.placeholder(tf.int32, )
    acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)
    rec, rec_op = tf.metrics.recall(labels=x, predictions=y)
    pre, pre_op = tf.metrics.precision(labels=x, predictions=y)
    f1,  f1_op  = tf.contrib.metrics.f1_score(labels=x, predictions=y)

    def f(x):
        if (x>0.5):
            return 1
        else:
            return 0
        
    predictions = np.array(list(map(f, predictions)))

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    v = sess.run(acc_op, feed_dict={x: test_labels,y: predictions}) # Accuracy.
    r = sess.run(rec_op, feed_dict={x: test_labels,y: predictions}) # Recall.
    p = sess.run(pre_op, feed_dict={x: test_labels,y: predictions}) # Precision.
    f = sess.run(f1_op,  feed_dict={x: test_labels,y: predictions}) # F1.

    print("accuracy: ", v)
    print("recall:   ", r)
    print("precision:", p)
    print("f1:       ", 2*p*r/(p+r))

def loss(history_dict):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def acc(history_dict):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
   
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()    
