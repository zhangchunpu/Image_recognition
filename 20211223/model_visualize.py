from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import graphviz
import pydot

model = load_model('my_model')

hidden_layer_model_1 = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
hidden_layer_model_2 = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d').output)
hidden_layer_model_3 = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
hidden_layer_model_4 = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_1').output)
hidden_layer_model_5 = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)
hidden_layer_model_6 = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_2').output)



for i in range(10):

    fig = plt.figure(figsize=(5,5))
    im = Image.open(f'test_data/test_{i}.jpg')
    im = im.resize((128, 128))
    im = im.convert('L')
    im = np.array(im)
    im[im < 128] = 1
    im[im >= 128] = 0

    #中間層の出力をみる
    target = im.reshape((1, 128, 128, 1))
    hidden_output_1 = hidden_layer_model_1.predict(target)
    print(hidden_output_1.shape)
    output_1 = hidden_output_1.sum(axis=3)
    output_1 = output_1.reshape(126, 126)

    #
    hidden_output_2 = hidden_layer_model_2.predict(target)
    print(hidden_output_2.shape)
    output_2 = hidden_output_2.sum(axis=3)
    output_2 = output_2.reshape(63, 63)

    #
    hidden_output_3 = hidden_layer_model_3.predict(target)
    print(hidden_output_3.shape)
    output_3 = hidden_output_3.sum(axis=3)
    output_3 = output_3.reshape(61, 61)

    #
    hidden_output_4 = hidden_layer_model_4.predict(target)
    print(hidden_output_4.shape)
    output_4 = hidden_output_4.sum(axis=3)
    output_4 = output_4.reshape(30, 30)

    #
    hidden_output_5 = hidden_layer_model_5.predict(target)
    print(hidden_output_5.shape)
    output_5 = hidden_output_5.sum(axis=3) / 32
    output_5 = output_5.reshape(28, 28)

    #
    hidden_output_6 = hidden_layer_model_6.predict(target)
    print(hidden_output_6.shape)
    output_6 = hidden_output_6.sum(axis=3) / 32
    output_6 = output_6.reshape(14, 14)

    for i in range(6):
        ax = fig.add_subplot(2,3,i+1)
        if i == 0:
            ax.imshow(output_1)
        elif i ==1:
            ax.imshow(output_2)
        elif i ==2:
            ax.imshow(output_3)
        elif i ==3:
            ax.imshow(output_4)
        elif i ==4:
            ax.imshow(output_5)
        else:
            ax.imshow(output_6)

    fig.show()

