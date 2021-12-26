from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = load_model('my_model')

for i in range(10):
    im = Image.open(f'test_data/test_{i}.jpg')
    im = im.resize((128, 128))
    gray_img = im.convert('L')
    gray_img_np = np.array(gray_img)
    gray_img_np[gray_img_np < 128] = 1
    gray_img_np[gray_img_np >= 128] = 0

    plt.set_cmap("binary")
    fig = plt.figure()
    # # 1データずつ取得して手書き数字データを表示
    ax = fig.add_subplot(111) # 表示位置設定
    ax.imshow(gray_img_np)# 手書き数字データを表示
    fig.show()
    print(np.argmax(model.predict(gray_img_np.reshape((1,128,128,1)))))

