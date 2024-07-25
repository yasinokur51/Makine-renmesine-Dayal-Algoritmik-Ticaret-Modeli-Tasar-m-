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
# Popülasyon oluşturma
def generate_population(population_size, gene_size):
    population = []
    for _ in range(population_size):
        OPTİMİZERS = np.random.choice(['adam','nadam'])
        
        #EPOCHS = np.random.randint(10,30)
        EPOCHS = np.random.choice([50,55,60,65,70,75,80])
        #BATCHSİZE = np.random.randint(1, 32)
        BATCHSİZE = np.random.choice([4,8,12,16,20,26,32,40,48,56,64])
        #NÖRON = np.random.randint(5, 20)
        NÖRON = np.random.choice([50,55,60,65,70,75,80,85,90])
        #TİMESTAPS= np.random.randint(10, 50)
        TİMESTAPS= np.random.choice([15,20,25,30,35,40,45,50])
        chromosome = [TİMESTAPS,NÖRON,EPOCHS,BATCHSİZE,OPTİMİZERS]
        population.append(chromosome)
    return population

# Uygunluk fonksiyonu
def fitness(individual):
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
    data['<close_open_fark>'] = (data['<CLOSE>'].shift(1) - data['<OPEN>'])
    data = data.dropna()

    #timesteps =random.choice([10,15,20,25,30,35,40,45,50,55,60])
    timesteps=individual[0]
    print("TİMESTAP",timesteps)
    #Fnöron=random.choice([50,55,60,65,70,75,80,85,90,95,100])
    nöron=individual[1]
    print("NÖRON SAYISI",nöron)
    #epoch=random.choice([50,55,60,65,70,75,80,85,90,95,100])
    epoch=individual[2]
    print("EPOCHS",epoch)
    #batch_siz=random.choice([1,4,8,12,16,24,32])
    batch_siz=individual[3]
    print("BATCH_SİZE",batch_siz)

    # Özellikler ve etiketler için veri setlerini oluştur
    features = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>','<close_open_fark>']
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
    model.compile(loss='mean_squared_error', optimizer=individual[4])

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
    
    return test_score
    
# Seçim işlemi
def selection(population, fitness_values):
    selected_indices = random.choices(range(len(population)), weights=fitness_values, k=len(population))
    return [population[i] for i in selected_indices]

# Çaprazlama işlemi
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutasyon işlemi
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# Genetik algoritma
def genetic_algorithm(population_size, gene_size, generations, mutation_rate):
    population = generate_population(population_size, gene_size)
    for generation in range(generations):
        fitness_values = []
        for kromozom in population:
            #print(kromozom)
            fitness_values.append(fitness(kromozom))
        #print(fitness_values)
        selected_population = selection(population, fitness_values)
        
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    
    # En iyi bireyi seçme
    best_individual = max(population, key=fitness)
    return best_individual, fitness(best_individual)

data = pd.read_csv("MAYIS_AY.csv", sep='\t', encoding="utf-8")
data.drop('<VOL>', axis=1, inplace=True)
data.drop('<SPREAD>', axis=1, inplace=True)
data.drop('<TICKVOL>',axis=1,inplace=True)

# Kullanım örneği
population_size = 5
gene_size = 5
generations = 5
mutation_rate = 0.00001

best_individual, best_fitness = genetic_algorithm(population_size, gene_size, generations, mutation_rate)
#sonuçekranı(best_individual)
print("En iyi birey:", best_individual,"En düşük MSE:",best_fitness)