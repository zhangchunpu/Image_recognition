# 手書き数字のデータセット取得と手書きデータの表示
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import get_model_2
import matplotlib.pyplot as plt
import math
import numpy as np
import time



# MNISTデータを取得してnumpyの配列型に変換
mnist_x, mnist_y = fetch_openml('mnist_784', version=1, data_home="sklearn_MNIST_data", return_X_y=True)
list_mnist_x = np.array(mnist_x)
list_mnist_y = np.array(mnist_y)

# 全部のデータを表示すると時間がかかるので
# 最初の4データのみ表示
digits = list_mnist_x[0:4]

# 手書き数字(8×8ピクセル)を表示
plt.set_cmap("binary")
fig = plt.figure()
row_and_col = math.ceil(math.sqrt(len(digits)))

# データずつ取得して手書き数字データを表示
for i,item in enumerate(digits):
    ax = fig.add_subplot(row_and_col,row_and_col,i+1) # 表示位置設定
    ax.imshow(digits[i].reshape(28,28))# 手書き数字データを表示
    fig.show()

#traing test
X_train, X_test, y_train, y_test = train_test_split(list_mnist_x, list_mnist_y, test_size=0.5)
print(X_train.size, X_test.size, sep='\n')

#サイズの調節
# 28x28x1のサイズへ変換しましょう
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

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
# model = get_model_2()
# start_time = time.time()
# model.fit(X_train, y_train, epochs=10)
#
# #test dataで正解率を検証
# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
# print(loss_and_metrics)
#
# #モデルをpickleファイルに落とし込む
# model.save('my_model')
#
# #時間を計算する
# time_used = time.time()-start_time
# print(f'{np.round(time_used, 2)}s used for training')





