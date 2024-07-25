import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


"""
İNDİKATÖRLER
"""
"""
BUY TNRSI KODU
"""
def buy_interval_calculate_tnrsi(data, column_name='<CLOSE>', period=16):
    fiyatort=0
    Xort=(len(data)+1)/2
    Xort+=0.1
    toplami=0
    toplam=0
    B1i=[]
    B1=0
    B0i=0
    trendi=[]
    trendfarki=[]
    result_list=[]
    negatiffark=[]
    pozitiffark=[]
    for i in range(len(data)):
        toplami+=1
        toplam+=data['<CLOSE>'][i]
        fiyatort=toplam/toplami
        fiyat=(data['<CLOSE>'][i])
        B1i.append((fiyat-fiyatort)/(i-Xort))
        B1=sum(B1i)/len(B1i)
        B0i=fiyatort-(B1*Xort)
        trendi.append((B1*i)+B0i)
        trendfarki.append((data['<CLOSE>'][i])-trendi[i])
        if i==0:
            result_list.append(trendfarki[0])
        elif i==len(data):
            continue
        else:
            result_list.append(trendfarki[i]-trendfarki[i-1])
        
        if result_list[i]<0:
            negatiffark.append(i)
        else:
            negatiffark.append(0)
            
        if result_list[i]>=0:
            pozitiffark.append(i)
        else:
            pozitiffark.append(0)
    
    data['Buy_Daily Return'] = data[column_name].pct_change()

    # Artan ve azalan getirileri ayır
    data['Buy_Gain'] = pozitiffark
    data['Buy_Loss'] = negatiffark

    # Ortalama artan ve azalan getirileri hesapla
    avg_gain = data['Buy_Gain'].rolling(window=period).mean()
    avg_loss = data['Buy_Loss'].rolling(window=period).mean()

    # RSI hesapla
    TNRS = avg_gain / abs(avg_loss)
    Buy_TNRSİ = 100 - (100 / (1 + TNRS))
    data['BUY TNRSI']=Buy_TNRSİ
    return data
"""
SELL TNRSI KODU
"""
def sell_interval_calculate_tnrsi(data, column_name='<CLOSE>',period=8):
    fiyatort=0
    Xort=(len(data)+1)/2
    Xort+=0.1
    toplami=0
    toplam=0
    B1i=[]
    B1=0
    B0i=0
    trendi=[]
    trendfarki=[]
    result_list=[]
    negatiffark=[]
    pozitiffark=[]
    for i in range(len(data)):
        toplami+=1
        toplam+=data['<CLOSE>'][i]
        fiyatort=toplam/toplami
        fiyat=(data['<CLOSE>'][i])
        B1i.append((fiyat-fiyatort)/(i-Xort))
        B1=sum(B1i)/len(B1i)
        B0i=fiyatort-(B1*Xort)
        trendi.append((B1*i)+B0i)
        trendfarki.append((data['<CLOSE>'][i])-trendi[i])
        if i==0:
            result_list.append(trendfarki[0])
        elif i==len(data):
            continue
        else:
            result_list.append(trendfarki[i]-trendfarki[i-1])
        
        if result_list[i]<0:
            negatiffark.append(i)
        else:
            negatiffark.append(0)
            
        if result_list[i]>=0:
            pozitiffark.append(i)
        else:
            pozitiffark.append(0)

    data['Sell_Daily Return'] = data[column_name].pct_change()

    # Artan ve azalan getirileri ayır
    data['Sell_Gain'] = pozitiffark
    data['Sell_Loss'] = negatiffark

    # Ortalama artan ve azalan getirileri hesapla
    avg_gain = data['Sell_Gain'].rolling(window=period).mean()
    avg_loss = data['Sell_Loss'].rolling(window=period).mean()

    # RSI hesapla
    TNRS = avg_gain / abs(avg_loss)
    Sell_TNRSİ = 100 - (100 / (1 + TNRS))
    data['SELL TNRSI']=Sell_TNRSİ
    return data
"""
BOLİNGER BAND KODU
"""
def calculate_bollinger_bands(data, window_size=14, num_std_dev=2,sma_period=14):
    # Basit Hareketli Ortalama (SMA) hesapla
    data['SMA'] = data['<CLOSE>'].rolling(window=sma_period).mean()

    # Güncel SMA'nın geçmiş değerlerle karşılaştırılması
    data['Trend'] = 'Unknown'
    data.loc[data['SMA'] > data['SMA'].shift(1), 'Trend'] = 'Up'
    data.loc[data['SMA'] < data['SMA'].shift(1), 'Trend'] = 'Down'
    # Standart sapma hesapla
    data['STD'] = data['<CLOSE>'].rolling(window=window_size).std()
    
    # Üst bant ve alt bant hesapla
    data['UpperBand'] = data['SMA'] + (num_std_dev * data['STD'])
    data['LowerBand'] = data['SMA'] - (num_std_dev * data['STD'])
    BBilist=[]
    for i in range(0,len(data['<CLOSE>'])):
        fiyat=(data['<CLOSE>'][i])
        BBi=(fiyat-data['LowerBand'][i])/(data['UpperBand'][i]-data['LowerBand'][i])
        BBilist.append(BBi)
    data['BBi']=BBilist
    return data
"""
ROC KODU
"""
def calculate_roc(data, n=14):
    # Fiyat değişimini hesapla
    data['PriceChange'] = data['<CLOSE>'].diff(n)
    
    # ROC hesapla
    data['ROC'] = (data['PriceChange'] / data['<CLOSE>'].shift(n)) * 100
    
    return data
"""
SO KODU
"""
def calculate_stochastic_oscillator(data, k_period=14):
    # En yüksek ve en düşük fiyatları belirle
    data['High_Max'] = data['<HIGH>'].rolling(window=k_period).max()
    data['Low_Min'] = data['<LOW>'].rolling(window=k_period).min()
    
    # Stokastik Osilatörü hesapla
    data['SO'] = ((data['<CLOSE>'] - data['Low_Min']) / (data['High_Max'] - data['Low_Min'])) * 100
    
    return data
"""
CCI KODU
"""
def calculate_commodity_channel_index(data, window_size=14):
    
    # Tipik Fiyatın hareketli ortalama üzerindeki sapmaları hesapla
    data['CCI'] = (data['<CLOSE>'] - data['SMA']) / (0.015 * data['<CLOSE>'].rolling(window=window_size).std())
    
    return data
"""
PPO KODU
"""
def calculate_ppo(data, short_window=12, long_window=26):
    # Kısa vadeli ve uzun vadeli hareketli ortalamaları hesapla
    data['Short_MA'] = data['<CLOSE>'].rolling(window=short_window).mean()
    data['Long_MA'] = data['<CLOSE>'].rolling(window=long_window).mean()
    
    # PPO hesapla
    data['PPO'] = ((data['Short_MA'] - data['Long_MA']) / data['Long_MA']*100)
    
    return data
"""
AROON KODU
"""
def calculate_aroon_oscillator(data, window=14):
    high = data['<HIGH>']
    low = data['<LOW>']

    # Aroon Up
    aroon_up = high.rolling(window=window).apply(lambda x: x.argmax(), raw=True) / window * 100

    # Aroon Down
    aroon_down = low.rolling(window=window).apply(lambda x: x.argmin(), raw=True) / window * 100

    # Aroon Oscillator
    aroon_oscillator = aroon_up - aroon_down
    data['AROON']=aroon_oscillator
    return data


"""
İNDİKATÖR SİNYALLERİ
"""
# TAMAM SON 8 TNRSI DAN EN AZ 3 BUY SİNYALİ İSE OKEY
def tnrsı_signals(data, overbought_threshold=62, oversold_threshold=44):
    signals=[]
    for i in range(0, len(data)):
        if data['BUY TNRSI'][i] < oversold_threshold:
                signals.append("BUY")
        elif data['SELL TNRSI'][i] > overbought_threshold:
                signals.append("SELL")
        else:
            signals.append("HOLD")
    data['tnrsı_signal']=signals
    return signals
# son 14 sinyalda 1 tane bile BUY varsa okey
def bbi_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['<CLOSE>'][i]<data['LowerBand'][i]:
            signals.append("BUY")
        elif data['<CLOSE>'][i]>data['UpperBand'][i]:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['bbi_signal']=signals
    return data
# ROC BASİT NORMAL SİNYALE GÖRE AL SAT YAPIYOR
def roc_signal(data):
    signals=[]
    for i in range(len(data)):
        # Al-Sat sinyallerini oluşturma
        if data['ROC'][i] <0:
            signals.append("BUY")
        elif data['ROC'][i] >0:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['roc_signal']=signals
    return data            
# AL SAT BU DAHİL EDİLMEYECEK
def so_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['SO'][i]<20 and data['SO'][i-1]>=20:
            signals.append("BUY")
        elif data['SO'][i]<80 and data['SO'][i-1]>=80:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['so_signal']=signals
    return data
# SON 14 DEĞER TOPLAMI EKSİ İSE  
def ccı_signal(data, window_size=14, num_std_dev=2,sma_period=14):
    signals=[]
    for i in range(len(data)):
        if data['CCI'][i]<-100:
            signals.append("BUY")
        elif data['CCI'][i]>100:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['ccı_signal']=signals
    return data
# AL SAT BU DAHİL EDİLMEYECEK
def ppo_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['PPO'][i]<0:
            signals.append("BUY")
        elif data['PPO'][i]>0:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['ppo_signal']=signals
    return data

# SON 5 TANEDEN 3 İSE OKEY
def aroon_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['AROON'][i]<0 :
            signals.append("BUY")
        elif data['AROON'][i]>0 :
            signals.append("SELL")
        else:
            signals.append("HOLD")
    data['aroon_signal']=signals
    return data

# SON 5 TANEDEN 3 İSE OKEY
def calculate_ichimoku_signal(data):
    high = data['<HIGH>']
    low = data['<LOW>']
    close = data['<CLOSE>']

    # Tenkan Sen (Dönüm Çizgisi)
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

    # Kijun Sen (Taban Çizgisi)
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

    # Senkou Span A (A Gölgesi)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (B Gölgesi)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)

    # Chikou Span (Geri Çekilme Çizgisi)
    chikou_span = close.shift(-26)
    
    # Al-Sat Sinyallerini Oluştur
    Tenkan_Kijun_Cross = np.where(tenkan_sen > kijun_sen, 1, 0)
    Price_Above_Below_Cloud = np.where(data['<CLOSE>'] > senkou_span_a, 1, -1)
    
    # Al-Sat Sinyalleri için genel sinyal
    ichimoku = np.where((Tenkan_Kijun_Cross == 1) & (Price_Above_Below_Cloud == 1), 1,
                      np.where((Tenkan_Kijun_Cross == 0) & (Price_Above_Below_Cloud == -1), -1, 0))
    
    ichimoku_signal = np.where((Tenkan_Kijun_Cross == 1) & (Price_Above_Below_Cloud == 1), "SELL",
                      np.where((Tenkan_Kijun_Cross == 0) & (Price_Above_Below_Cloud == -1), "BUY", "HOLD"))
  
    data['ichimoku'] = ichimoku
    data['ichimoku_signal']=ichimoku_signal
    return data

def tahmin_dip_seattlepoint(data):
    seattlepoint=[]
    signals=[]
    for i in range(len(data)):
        if i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6:
            signals.append("?")
            continue
        else:
            dip=i-3
            if data['<CLOSE>'][dip]<data['<CLOSE>'][i-4] and data['<CLOSE>'][dip]<data['<CLOSE>'][i-5] and data['<CLOSE>'][dip]<data['<CLOSE>'][i-6] and data['<CLOSE>'][dip]<data['<CLOSE>'][i-2] and data['<CLOSE>'][dip]<data['<CLOSE>'][i-1] and data['<CLOSE>'][dip]<data['<CLOSE>'][i]:
                seattlepoint.append(dip)
                signals.append("?")
                signals[-4]="DİP"
            else:
                signals.append("?")
                continue
    data['DİPSİGNAL']=signals
    return data

def tahmin_tavan_seattlepoint(data):
    seattlepoint=[]
    signals=[]
    for i in range(len(data)):
        if i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6:
            signals.append("?")
            continue
        else:
            tavan=i-3
            if data['<CLOSE>'][tavan]>data['<CLOSE>'][i-4] and data['<CLOSE>'][tavan]>data['<CLOSE>'][i-5] and data['<CLOSE>'][tavan]>data['<CLOSE>'][i-6] and data['<CLOSE>'][tavan]>data['<CLOSE>'][i-2] and data['<CLOSE>'][tavan]>data['<CLOSE>'][i-1] and data['<CLOSE>'][tavan]>data['<CLOSE>'][i]:
                seattlepoint.append(tavan)
                signals.append("?")
                signals[-4]="TAVAN"
            else:
                signals.append("?")
                continue
    data['TAVANSİGNAL']=signals
    return data


def real_al_sat(data,initial_investment=100,transaction_cost=0.001):
    firstcapital=initial_investment
    capital = initial_investment
    position = 0  # Pozisyon durumu: 0 = beklemede, 1 = uzun pozisyon, -1 = kısa pozisyon
    shares_bought=0
    global alımfiyatları,satımfiyatları
    alımfiyatları=[]
    satımfiyatları=[]    
    signals=[]
    real_signal=[]
    for i in range(0,len(data)):
        if i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7 or i==8 or i==9 or i==10 or i==11 or i==12 or i==13 or i==14 or i==15 or i==16 or i==17:
            signals.append("?")
            real_signal.append("---")
            continue
        else:
            tnrsı_signals=[data['tnrsı_signal'][i-16],data['tnrsı_signal'][i-15],data['tnrsı_signal'][i-14],data['tnrsı_signal'][i-13],data['tnrsı_signal'][i-12],data['tnrsı_signal'][i-11],data['tnrsı_signal'][i-10],data['tnrsı_signal'][i-9], data['tnrsı_signal'][i-8], data['tnrsı_signal'][i-7],data['tnrsı_signal'][i-6], data['tnrsı_signal'][i-5], data['tnrsı_signal'][i-4], data['tnrsı_signal'][i-3]]
            bbi_signals=[data['bbi_signal'][i-16], data['bbi_signal'][i-15], data['bbi_signal'][i-14], data['bbi_signal'][i-13], data['bbi_signal'][i-12], data['bbi_signal'][i-11], data['bbi_signal'][i-10], data['bbi_signal'][i-9], data['bbi_signal'][i-8], data['bbi_signal'][i-7], data['bbi_signal'][i-6], data['bbi_signal'][i-5], data['bbi_signal'][i-4], data['bbi_signal'][i-3]]   
            so_signals=[data['so_signal'][i-16],data['so_signal'][i-15],data['so_signal'][i-14],data['so_signal'][i-13],data['so_signal'][i-12],data['so_signal'][i-11],data['so_signal'][i-10],data['so_signal'][i-9], data['so_signal'][i-8], data['so_signal'][i-7],data['so_signal'][i-6], data['so_signal'][i-5], data['so_signal'][i-4], data['so_signal'][i-3]]
            roc_signals=[data['roc_signal'][i-16],data['roc_signal'][i-15],data['roc_signal'][i-14],data['roc_signal'][i-13],data['roc_signal'][i-12],data['roc_signal'][i-11],data['roc_signal'][i-10],data['roc_signal'][i-9], data['roc_signal'][i-8], data['roc_signal'][i-7],data['roc_signal'][i-6], data['roc_signal'][i-5], data['roc_signal'][i-4], data['roc_signal'][i-3]]
            ppo_signals=[data['ppo_signal'][i-16],data['ppo_signal'][i-15],data['ppo_signal'][i-14],data['ppo_signal'][i-13],data['ppo_signal'][i-12],data['ppo_signal'][i-11],data['ppo_signal'][i-10],data['ppo_signal'][i-9], data['ppo_signal'][i-8], data['ppo_signal'][i-7],data['ppo_signal'][i-6], data['ppo_signal'][i-5], data['ppo_signal'][i-4], data['ppo_signal'][i-3]]
            aroon_signals=[data['aroon_signal'][i-16],data['aroon_signal'][i-15],data['aroon_signal'][i-14],data['aroon_signal'][i-13],data['aroon_signal'][i-12],data['aroon_signal'][i-11],data['aroon_signal'][i-10],data['aroon_signal'][i-9], data['aroon_signal'][i-8], data['aroon_signal'][i-7],data['aroon_signal'][i-6], data['aroon_signal'][i-5], data['aroon_signal'][i-4], data['aroon_signal'][i-3]]
            ichimoku_signals=[data['ichimoku_signal'][i-16],data['ichimoku_signal'][i-15],data['ichimoku_signal'][i-14],data['ichimoku_signal'][i-13],data['ichimoku_signal'][i-12],data['ichimoku_signal'][i-11],data['ichimoku_signal'][i-10],data['ichimoku_signal'][i-9], data['ichimoku_signal'][i-8], data['ichimoku_signal'][i-7],data['ichimoku_signal'][i-6], data['ichimoku_signal'][i-5], data['ichimoku_signal'][i-4], data['ichimoku_signal'][i-3]]
            ccı_signals=[data['ccı_signal'][i-16], data['ccı_signal'][i-15], data['ccı_signal'][i-14], data['ccı_signal'][i-13], data['ccı_signal'][i-12], data['ccı_signal'][i-11], data['ccı_signal'][i-10], data['ccı_signal'][i-9], data['ccı_signal'][i-8], data['ccı_signal'][i-7], data['ccı_signal'][i-6], data['ccı_signal'][i-5], data['ccı_signal'][i-4], data['ccı_signal'][i-3]]   
            
            if data['DİPSİGNAL'][i-3]=="DİP" and [(ccı_signals.count("BUY")/len(ccı_signals)>=0.2) and (tnrsı_signals.count("BUY")/len(tnrsı_signals)>=0.44) and (roc_signals.count("BUY")/len(roc_signals)>=0.2) and (ppo_signals.count("BUY")>=len(ppo_signals)>=0.9) and (aroon_signals.count("BUY")>=len(aroon_signals)>=0.40) and (ichimoku_signals.count("BUY")>=len(ichimoku_signals)>=0.43)]:
                signals.append("AL")
                #print("BUY",i)
                if ((abs(data['<CLOSE>'][i-3] - data['Low_prediction'][i-3]) / max(data['<CLOSE>'][i-3], data['Low_prediction'][i-3]) * 100)<0.1).all():
                    real_signal.append("REAL BUY")
                    #print("REAL BUY",i)
                    if position in [0,-1]:
                        print("ALACAĞIM","---","Alım Fiyatı:","---",data['<CLOSE>'][i])
                        print("Capital:",capital)
                        position=1
                        capital-=capital*transaction_cost
                        shares_bought=capital/(data['<CLOSE>'][i])
                        capital=0
                        print("ALDIM","Capital:",capital,"ALDIĞIM YER",i,"Sharesbought:",shares_bought)
                        print("----------------------------")
                        alımfiyatları.append(data['<CLOSE>'][i])
                else:
                    real_signal.append("---")   
            elif data['TAVANSİGNAL'][i-3]=="TAVAN" and [(ccı_signals.count("SELL")/len(ccı_signals)>=0.2) and (tnrsı_signals.count("SELL")/len(tnrsı_signals)>=0.44) and (roc_signals.count("SELL")/len(roc_signals)>=0.2) and (ppo_signals.count("SELL")>=len(ppo_signals)>=0.9) and (aroon_signals.count("SELL")>=len(aroon_signals)>=0.40) and (ichimoku_signals.count("SELL")>=len(ichimoku_signals)>=0.43)]:
                signals.append("SELL")
                #print("SELL",i)
                if ((abs(data['<CLOSE>'][i-3] - data['High_prediction'][i-3]) / max(data['<CLOSE>'][i-3], data['High_prediction'][i-3]) * 100)<0.1).all():
                    real_signal.append("REAL SELL")
                    #print("REAL SELL",i)
                    if position==1 and alımfiyatları[-1]<data['<CLOSE>'][i] :
                        print("SATACAĞIM","---","Satım Fiyatı:","---",data['<CLOSE>'][i])
                        print("Capital:",capital)
                        position=-1
                        capital+=shares_bought*data['<CLOSE>'][i]
                        capital-=capital*transaction_cost
                        shares_bought=0
                        print("SATTIM","Capital:",capital,"SATTIĞIM YER",i,"Sharesbought:",shares_bought)
                        print("----------------------------")
                        satımfiyatları.append(data['<CLOSE>'][i])
                else:
                    real_signal.append("---")
            else:
                signals.append("---")
                real_signal.append("---")   
                continue
    data['AL_SAT']=signals
    data['REAL_AL_SAT']=real_signal
    return data

"""
ÖRNEK VERİ
"""
data= pd.read_csv("MAYIS_22GÜN.csv",sep='\t', encoding="utf-8")
#
#data2=pd.read_csv("ASK_BİD_GBPUSD_202311080000_202311082359.csv",sep='\t', encoding="utf-8")
"""
PREDİCTİON
"""
predictions = pd.read_csv('next_day_predictions.csv')

# Tarih sütunlarını datetime formatına çevir
data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y.%m.%d')
predictions['Date'] = pd.to_datetime(predictions['Date'], format='%Y.%m.%d')

# Tahmin sütunlarını eklemek için boş sütunlar oluştur
data['Open_prediction'] = None
data['High_prediction'] = None
data['Low_prediction'] = None
data['Close_prediction'] = None

# Tarihler eşleştiğinde tahmin verilerini ekle
for i, row in data.iterrows():
    target_date = row['<DATE>']
    if target_date in predictions['Date'].values:
        prediction_row = predictions.loc[predictions['Date'] == target_date]
        data.at[i, 'Open_prediction'] = prediction_row['Open'].values[0]
        data.at[i, 'High_prediction'] = prediction_row['High'].values[0]
        data.at[i, 'Low_prediction'] = prediction_row['Low'].values[0]
        data.at[i, 'Close_prediction'] = prediction_row['Close'].values[0]

# İNDİKATÖR HESAP VE SİNYAL
buy_interval_calculate_tnrsi(data)
sell_interval_calculate_tnrsi(data)
tnrsı_signals(data)
calculate_bollinger_bands(data)
bbi_signal(data)
calculate_roc(data)
roc_signal(data)
calculate_stochastic_oscillator(data)
so_signal(data)
calculate_commodity_channel_index(data)
ccı_signal(data)
calculate_ppo(data)
ppo_signal(data)
calculate_aroon_oscillator(data)
aroon_signal(data)
calculate_ichimoku_signal(data)
tahmin_dip_seattlepoint(data)
tahmin_tavan_seattlepoint(data)
real_al_sat(data)
data = data.iloc[25:]

"""
SUTÜN SİLME
"""
data.drop(['Short_MA','Long_MA','PriceChange','LowerBand','UpperBand','STD','Trend','SMA','High_Max','Low_Min','Sell_Daily Return','Buy_Daily Return','Buy_Gain','Sell_Gain','Sell_Loss','Buy_Loss'],axis=1,inplace=True)
