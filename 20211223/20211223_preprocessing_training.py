# 手書き数字のデータセット取得と手書きデータの表示
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import get_model_3
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from PIL import Image



# MNISTデータを取得してnumpyの配列型に変換
mnist_x, mnist_y = fetch_openml('mnist_784', version=1, data_home="sklearn_MNIST_data", return_X_y=True)
list_mnist_x = np.array(mnist_x)
list_mnist_y = np.array(mnist_y)

# 手書き数字(8×8ピクセル)を表示
# plt.set_cmap("binary")
# fig = plt.figure()
# ax = fig.add_subplot(111)

# データずつ取得して手書き数字データを表示
new_mnist_x = []
for i, item in enumerate(list_mnist_x):
    im_array = list_mnist_x[i].reshape(28, 28)
    im = Image.fromarray(im_array)
    im = im.resize((128, 128))
    new_im_array = np.array(im)
    new_mnist_x.append(new_im_array.reshape(16384))

for i, item in enumerate(list_mnist_x):
    im_array = list_mnist_x[i].reshape(28, 28)
    im = Image.fromarray(im_array)
    im = im.resize((128, 128))
    rotate_angle = np.random.randint(low=-10, high=10, size=None)
    im = im.rotate(rotate_angle, fillcolor=0)
    new_im_array = np.array(im)
    new_mnist_x.append(new_im_array.reshape(16384))

new_mnist_x=np.array(new_mnist_x)
new_mnist_y=np.append(list_mnist_y, list_mnist_y)

#traing test
X_train, X_test, y_train, y_test = train_test_split(new_mnist_x, new_mnist_y, test_size=0.5)

#サイズの調節
# 28x28x1のサイズへ変換しましょう
X_train = X_train.reshape(X_train.shape[0], 128, 128,1)
X_test = X_test.reshape(X_test.shape[0], 128, 128,1)
print(X_train.size, X_test.size, sep='\n')

#kerasで扱えるようにfloat32型に変換する
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#正規化
X_train /= 255
X_test /= 255

# ターゲットとなるyを変換
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# モデルの訓練（エポック １０）
model = get_model_3()
start_time = time.time()
model.fit(X_train, y_train, epochs=10)

#test dataで正解率を検証
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print(loss_and_metrics)

#モデルをpickleファイルに落とし込む
model.save('my_model')

#時間を計算する
time_used = time.time()-start_time
print(f'{np.round(time_used, 2)}s used for training')




