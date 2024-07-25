import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Nadam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import pywt
import matplotlib.pyplot as plt

def wavelet_denoise(data, column, wavelet='db1', level=1):
    signal = data[column].values
    coeffs = pywt.wavedec(signal, wavelet, mode='per')
    threshold = (np.median(np.abs(coeffs[-level])) / 0.6745) * (np.sqrt(2 * np.log(len(signal))))
    new_coeffs = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs]
    denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')
    # Uzunluk eşleşmiyorsa sinyali orijinal uzunluğa getir
    if len(denoised_signal) != len(signal):
        if len(denoised_signal) > len(signal):
            # Eğer denoised_signal daha uzunsa, fazladan değerleri kırp
            denoised_signal = denoised_signal[:len(signal)]
        else:
            # Eğer denoised_signal daha kısaysa, son değeri tekrarlayarak uzat
            extra_len = len(signal) - len(denoised_signal)
            denoised_signal = np.append(denoised_signal, [denoised_signal[-1]] * extra_len)
    return denoised_signal

def reshape_data(data, timesteps):
    reshaped_data = []
    for i in range(len(data) - timesteps + 1):
        reshaped_data.append(data[i:i + timesteps])
    return np.array(reshaped_data)


data = pd.read_csv("MAYIS_AY.csv", sep='\t', encoding="utf-8")
# 'Open', 'High', 'Low', ve 'Close' sütunlarında gürültü azaltma işlemi uygula

#data['<OPEN>'] = wavelet_denoise(data, '<OPEN>', wavelet='db1', level=1)
#data['<HIGH>'] = wavelet_denoise(data, '<HIGH>', wavelet='db1', level=1)
#data['<LOW>'] = wavelet_denoise(data, '<LOW>', wavelet='db1', level=1)
#data['<CLOSE>'] = wavelet_denoise(data, '<CLOSE>', wavelet='db1', level=1)

data['close_open_diff'] = (data['<CLOSE>'].shift(1) - data['<OPEN>'])
data = data.dropna()

#timesteps =random.choice([10,15,20,25,30,35,40,45,50,55,60])
timesteps=30
print("TİMESTAP",timesteps)
#Fnöron=random.choice([50,55,60,65,70,75,80,85,90,95,100])
nöron=75
print("NÖRON SAYISI",nöron)
#epoch=random.choice([50,55,60,65,70,75,80,85,90,95,100])
epoch=50
print("EPOCHS",epoch)
#batch_siz=random.choice([1,4,8,12,16,24,32])
batch_siz=12
print("BATCH_SİZE",batch_siz)

# Özellikler ve etiketler için veri setlerini oluştur
features = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>','close_open_diff']
labels = data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values

# Split the dataset
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Apply scaling
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_labels = MinMaxScaler(feature_range=(-1, 1))

train_data_scaled = scaler_features.fit_transform(train_data[features])
test_data_scaled = scaler_features.transform(test_data[features])
train_labels_scaled = scaler_labels.fit_transform(train_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']])
test_labels_scaled = scaler_labels.transform(test_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']])

# Reshaping for LSTM

train_x = reshape_data(train_data_scaled, timesteps)
test_x = reshape_data(test_data_scaled, timesteps)
train_y = train_labels_scaled[timesteps-1:]
test_y = test_labels_scaled[timesteps-1:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(nöron, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))  
model.add(LSTM(nöron))
model.add(Dropout(0.2))
model.add(Dense(4))
model.compile(loss='mean_squared_error', optimizer='nadam')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1, mode='min')

# Train the model
model.fit(train_x, train_y, epochs=epoch, batch_size=batch_siz, validation_data=(test_x, test_y), callbacks=[early_stopping])

# Predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Inverse transform predictions
train_predict = scaler_labels.inverse_transform(train_predict)
test_predict = scaler_labels.inverse_transform(test_predict)
    
# Model performance evaluation
train_score = math.sqrt(mean_squared_error(train_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values[timesteps-1:], train_predict))
test_score = math.sqrt(mean_squared_error(test_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values[timesteps-1:], test_predict))
print("Train Score:", train_score)
print("Test Score:", test_score)

# Son y veri için sonraki gün tahmini yap
y=-22
recent_data = test_x[y:]
next_day_preds = []

for i in range(len(recent_data)):
    next_day_pred = model.predict(recent_data[i].reshape(1, timesteps, len(features)))
    next_day_pred = scaler_labels.inverse_transform(next_day_pred)
    next_day_preds.append(next_day_pred[0])

# Tahminleri DataFrame'e dönüştür
next_day_preds_df = pd.DataFrame(next_day_preds, columns=['Open', 'High', 'Low', 'Close'])

# Tahmin edilen günlerin tarihlerini ekle
dates = data.iloc[train_size + timesteps - 1:]['<DATE>'].values[y:]
next_day_preds_df['Date'] = dates

print(next_day_preds_df)

recent_data_actual = test_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values[y:]
recent_data_pred = next_day_preds_df[['Open', 'High', 'Low', 'Close']].values

rmse = np.sqrt(mean_squared_error(recent_data_actual, recent_data_pred, multioutput='raw_values'))
print("Root Mean Squared Error for next day predictions on last y data points:", rmse)

next_day_preds_df.to_csv('next_day_predictions.csv', index=False)

# Grafik oluşturma
plt.figure(figsize=(15, 10))

# Açılış fiyatı (Open) karşılaştırması
plt.subplot(2, 2, 1)
plt.plot(recent_data_actual[:, 0], label='Actual Open')
plt.plot(recent_data_pred[:, 0], label='Predicted Open')
plt.title('Open Price Comparison')
plt.legend()

# Yüksek fiyat (High) karşılaştırması
plt.subplot(2, 2, 2)
plt.plot(recent_data_actual[:, 1], label='Actual High')
plt.plot(recent_data_pred[:, 1], label='Predicted High')
plt.title('High Price Comparison')
plt.legend()

# Düşük fiyat (Low) karşılaştırması
plt.subplot(2, 2, 3)
plt.plot(recent_data_actual[:, 2], label='Actual Low')
plt.plot(recent_data_pred[:, 2], label='Predicted Low')
plt.title('Low Price Comparison')
plt.legend()

# Kapanış fiyatı (Close) karşılaştırması
plt.subplot(2, 2, 4)
plt.plot(recent_data_actual[:, 3], label='Actual Close')
plt.plot(recent_data_pred[:, 3], label='Predicted Close')
plt.title('Close Price Comparison')
plt.legend()

plt.tight_layout()
plt.show()
