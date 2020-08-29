from keras.utils import np_utils
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras.callbacks
import tensorflow as tf
import numpy as np

#랜덤 seed 생성
np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]


X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0


# 훈련셋, 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 700, replace=False) # 중복선택 방지
val_rand_idxs = np.random.choice(10000, 300, replace=False)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]


# 라벨링 전환 -> one-hot vector로 변환
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)


# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 4. 모델 학습
# TensorBoard >> tensorboard --logdir=~/경로
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
                                      #profile_batch = 10000) #버전에 따라 관련 오류 발생시 파라미터 추가
# 학습 조기종료
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
hist = model.fit(X_train, Y_train, epochs=200, batch_size=10, validation_data=(X_val, Y_val), callbacks=[tb_hist, early_stopping])

# 5. 모델 학습 과정 표시하기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

## 버전에 따라 accuracy일 수 있습니다.
acc_ax.plot(hist.history['acc'], 'b', label='train acc') 
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 사용하기
xhat_idx = np.random.choice(X_test.shape[0], 5)
xhat = X_test[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(np.argmax(Y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))

# 6. 모델 저장하기
model.save('mnist_mlp_model.h5')

# model 이미지 파일로 저장 , 버전에 따라 파라미터가 다를 수 있습니다.
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB"
    #expand_nested=False
    #dpi=96,
)

'''
# pydot.Dot
tf.keras.utils.model_to_dot(
    model,
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    subgraph=False,
)
'''
