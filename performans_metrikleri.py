import pandas as pd
import random
import numpy as np

"""
TEKNİK GÖSTERGELER
"""
def tnrsi(data, column_name='<CLOSE>', period=14):
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
    
    data['Daily Return'] = data[column_name].pct_change()

    # Artan ve azalan getirileri ayır
    data['Gain'] = pozitiffark
    data['Loss'] = negatiffark

    # Ortalama artan ve azalan getirileri hesapla
    avg_gain = data['Gain'].rolling(window=period).mean()
    avg_loss = data['Loss'].rolling(window=period).mean()

    # RSI hesapla
    TNRS = avg_gain / abs(avg_loss)
    Buy_TNRSİ = 100 - (100 / (1 + TNRS))
    data['TNRSI']=Buy_TNRSİ
    return data
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
def calculate_roc(data, n=14):
    # Fiyat değişimini hesapla
    data['PriceChange'] = data['<CLOSE>'].diff(n)
    
    # ROC hesapla
    data['ROC'] = (data['PriceChange'] / data['<CLOSE>'].shift(n)) * 100
    
    return data
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
  
    data['ichimoku_signal'] = ichimoku
    #data['ichimoku_signal']=ichimoku_signal
    return data
def calculate_commodity_channel_index(data, window_size=14):
    
    # Tipik Fiyatın hareketli ortalama üzerindeki sapmaları hesapla
    data['CCI'] = (data['<CLOSE>'] - data['SMA']) / (0.015 * data['<CLOSE>'].rolling(window=window_size).std())
    
    return data
def calculate_ppo(data, short_window=12, long_window=26):
    # Kısa vadeli ve uzun vadeli hareketli ortalamaları hesapla
    data['Short_MA'] = data['<CLOSE>'].rolling(window=short_window).mean()
    data['Long_MA'] = data['<CLOSE>'].rolling(window=long_window).mean()
    
    # PPO hesapla
    data['PPO'] = ((data['Short_MA'] - data['Long_MA']) / data['Long_MA']*100)
    
    return data
def calculate_stochastic_oscillator(data, k_period=14):
    # En yüksek ve en düşük fiyatları belirle
    data['High_Max'] = data['<HIGH>'].rolling(window=k_period).max()
    data['Low_Min'] = data['<LOW>'].rolling(window=k_period).min()
    
    # Stokastik Osilatörü hesapla
    data['SO'] = ((data['<CLOSE>'] - data['Low_Min']) / (data['High_Max'] - data['Low_Min'])) * 100
    
    return data

"""
TEKNİK SİNYALLER
"""
def tnrsı_signals(data, overbought_threshold=62, oversold_threshold=44):
    signals=[]
    for i in range(0, len(data)):
        if data['TNRSI'][i] < oversold_threshold:
                signals.append(-1)
        elif data['TNRSI'][i] > overbought_threshold:
                signals.append(1)
        else:
            signals.append(0)
    data['tnrsı_signal']=signals
    return data
def bbi_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['<CLOSE>'][i]<data['LowerBand'][i]:
            signals.append(-1)
        elif data['<CLOSE>'][i]>data['UpperBand'][i]:
            signals.append(1)
        else:
            signals.append(0)
    data['bbi_signal']=signals
    return data
def roc_signal(data):
    signals=[]
    for i in range(len(data)):
        # Al-Sat sinyallerini oluşturma
        if data['ROC'][i] <0:
            signals.append(-1)
        elif data['ROC'][i] >0:
            signals.append(1)
        else:
            signals.append(0)
    data['roc_signal']=signals
    return data   
def so_signal(data):
    signals = []
    for i in range(1, len(data)):
        if data['SO'][i] < 20 and data['SO'][i-1] >= 20:
            signals.append(-1)
        elif data['SO'][i] > 80 and data['SO'][i-1] <= 80:
            signals.append(1)
        else:
            signals.append(0)
    data['so_signal'] = [0] + signals  # İlk değer için 0 ekliyoruz
    return data
def ccı_signal(data, window_size=14):
    signals=[]
    for i in range(len(data)):
        if data['CCI'][i]<-100:
            signals.append(-1)
        elif data['CCI'][i]>100:
            signals.append(1)
        else:
            signals.append(0)
    data['ccı_signal']=signals
    return data
def ppo_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['PPO'][i]<0:
            signals.append(-1)
        elif data['PPO'][i]>0:
            signals.append(1)
        else:
            signals.append(0)
    data['ppo_signal']=signals
    return data
def aroon_signal(data):
    signals=[]
    for i in range(len(data)):
        if data['AROON'][i]<0 :
            signals.append(-1)
        elif data['AROON'][i]>0 :
            signals.append(1)
        else:
            signals.append(0)
    data['aroon_signal']=signals
    return data

"""
PERFORMANS METRİKLERİ
"""
def calculate_signal_returns(data):
    """
    Sinyallere dayalı getirileri hesaplar.
    """
    data['next_close'] = data['<CLOSE>'].shift(-1)
    data['return'] = (data['next_close'] - data['<CLOSE>']) / data['<CLOSE>']
    
    indicators = ['tnrsı_signal', 'bbi_signal', 'aroon_signal', 'ichimoku_signal', 'ccı_signal', 'roc_signal', 'ppo_signal', 'so_signal']
    for indicator in indicators:
        signal_column = indicator
        data[signal_column + '_return'] = data[signal_column].shift(1) * data['return']
    return data
def calculate_sortino_ratio(data, signal_return_column, risk_free_rate=0.0):
    df = data[[signal_return_column]].dropna()
    df['excess_return'] = df[signal_return_column] - risk_free_rate
    negative_volatility = df[df['excess_return'] < 0]['excess_return'].std() * np.sqrt(252)
    annualized_return = df['excess_return'].mean() * 252
    if negative_volatility != 0:
        sortino_ratio = annualized_return / negative_volatility
    else:
        sortino_ratio = np.nan
    return sortino_ratio
def calculate_maximum_drawdown(data, signal_return_column):
    df = data[[signal_return_column]].dropna()
    cumulative_return = (1 + df[signal_return_column]).cumprod()
    peak = cumulative_return.expanding(min_periods=1).max()
    drawdown = (cumulative_return - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown
def calculate_calmar_ratio(data, signal_return_column):
    annualized_return = data[signal_return_column].mean() * 252
    max_drawdown = calculate_maximum_drawdown(data, signal_return_column)
    if max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = np.nan
    return calmar_ratio
def calculate_gain_loss_ratio(data, signal_return_column):
    df = data[[signal_return_column]].dropna()
    gains = df[df[signal_return_column] > 0][signal_return_column].mean()
    losses = -df[df[signal_return_column] < 0][signal_return_column].mean()
    if losses != 0:
        gain_loss_ratio = gains / losses
    else:
        gain_loss_ratio = np.nan
    return gain_loss_ratio
def calculate_sharpe_ratio(data, signal_return_column, risk_free_rate=0.0):
    df = data[[signal_return_column]].dropna()
    excess_return = df[signal_return_column] - risk_free_rate
    annualized_excess_return = excess_return.mean() * 252
    annualized_volatility = excess_return.std() * np.sqrt(252)
    if annualized_volatility != 0:
        sharpe_ratio = annualized_excess_return / annualized_volatility
    else:
        sharpe_ratio = np.nan
    return sharpe_ratio

def calculate_performance_metrics_for_chunks(data, chunk_size, risk_free_rate=0.00):
    """
    Her bir chunk için performans metriklerini hesaplar.
    """
    indicators = ['tnrsı_signal', 'aroon_signal', 'ichimoku_signal', 'ccı_signal', 'roc_signal', 'ppo_signal']#,'bbi_signal','so_signal']
    performance_results = {}
    
    for indicator in indicators:
        performance_results[indicator] = {'Sharpe Ratio': [], 'Sortino Ratio': [], 'Max Drawdown': [], 'Calmar Ratio': [], 'Gain/Loss Ratio': []}

        for start in range(0, len(data), chunk_size):
            end = start + chunk_size
            chunk = data.iloc[start:end].copy()
            
            if len(chunk) < chunk_size:
                continue
            
            chunk = calculate_signal_returns(chunk)
            signal_return_column = indicator + '_return'
            
            # Sharpe Oranını hesapla ve sakla
            sharpe_ratio = calculate_sharpe_ratio(chunk, signal_return_column, risk_free_rate)
            performance_results[indicator]['Sharpe Ratio'].append(sharpe_ratio)
            
            # Gain Loss hesapla ve sakla
            gain_loss = calculate_gain_loss_ratio(chunk, signal_return_column)            
            performance_results[indicator]['Gain/Loss Ratio'].append(gain_loss)
            
            # Calmar Oranını hesapla ve sakla
            calmar_ratio = calculate_calmar_ratio(chunk, signal_return_column)
            performance_results[indicator]['Calmar Ratio'].append(calmar_ratio)
            
            # Maksimum Çekilme Oranını hesapla ve sakla
            max_drawdown = calculate_maximum_drawdown(chunk, signal_return_column)
            performance_results[indicator]['Max Drawdown'].append(max_drawdown)
            
            # Sortino Oranını hesapla ve sakla
            sortino_ratio = calculate_sortino_ratio(chunk, signal_return_column)
            performance_results[indicator]['Sortino Ratio'].append(sortino_ratio)
    
    return performance_results

# Performans metriklerini normalize etmeden doğrudan sıralama fonksiyonları
def rank_indicators_by_metric(metrics):
    sorted_indicators = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    return {indicator: rank + 1 for rank, (indicator, _) in enumerate(sorted_indicators)}

# Gösterge sıralamasını hesaplamak için ana fonksiyon
def rank_indicators(performance_metrics):
    sharpe_ratios = {indicator: values['Sharpe Ratio'][-1] for indicator, values in performance_metrics.items() if values['Sharpe Ratio']}
    sortino_ratios = {indicator: values['Sortino Ratio'][-1] for indicator, values in performance_metrics.items() if values['Sortino Ratio']}
    max_drawdowns = {indicator: values['Max Drawdown'][-1] for indicator, values in performance_metrics.items() if values['Max Drawdown']}
    calmar_ratios = {indicator: values['Calmar Ratio'][-1] for indicator, values in performance_metrics.items() if values['Calmar Ratio']}
    gain_loss_ratios = {indicator: values['Gain/Loss Ratio'][-1] for indicator, values in performance_metrics.items() if values['Gain/Loss Ratio']}
    
    # Sıralamalar
    sharpe_rank = rank_indicators_by_metric(sharpe_ratios)
    sortino_rank = rank_indicators_by_metric(sortino_ratios)
    max_drawdown_rank = rank_indicators_by_metric(max_drawdowns)
    calmar_rank = rank_indicators_by_metric(calmar_ratios)
    gain_loss_rank = rank_indicators_by_metric(gain_loss_ratios)
    
    # Toplam sıralama puanlarının hesaplanması
    total_scores = {indicator: 0 for indicator in performance_metrics.keys()}
    for indicator in total_scores.keys():
        total_scores[indicator] += sharpe_rank.get(indicator, len(performance_metrics))
        total_scores[indicator] += sortino_rank.get(indicator, len(performance_metrics))
        total_scores[indicator] += max_drawdown_rank.get(indicator, len(performance_metrics))
        total_scores[indicator] += calmar_rank.get(indicator, len(performance_metrics))
        total_scores[indicator] += gain_loss_rank.get(indicator, len(performance_metrics))
    
    # Genel sıralama
    sorted_total_scores = sorted(total_scores.items(), key=lambda x: x[1])
    
    return sorted_total_scores, sharpe_rank, sortino_rank, max_drawdown_rank, calmar_rank, gain_loss_rank


data = pd.read_csv("MAYIS_AY.csv", sep='\t', encoding="utf-8")
tnrsi(data)
calculate_bollinger_bands(data)
calculate_aroon_oscillator(data)
calculate_commodity_channel_index(data)
calculate_ichimoku_signal(data)
calculate_roc(data)
calculate_ppo(data)
calculate_stochastic_oscillator(data)
tnrsı_signals(data)
bbi_signal(data)
aroon_signal(data)
ccı_signal(data)
roc_signal(data)
ppo_signal(data)
so_signal(data)
data.drop(['<SPREAD>', '<TICKVOL>', '<VOL>', 'Gain', 'Loss', 'Daily Return','LowerBand','UpperBand','SMA','Trend','STD','PriceChange','Short_MA','Long_MA','High_Max','Low_Min'], axis=1, inplace=True)
#data = data.iloc[25:]

x=22
# Performans metriklerini hesaplayan fonksiyonun çağrılması
performance_metrics = calculate_performance_metrics_for_chunks(data,x)

# Sıralama hesaplaması
ranked_indicators, sharpe_rank, sortino_rank, max_drawdown_rank, calmar_rank, gain_loss_rank = rank_indicators(performance_metrics)
"""
print("SON AY DEĞERLERİ")
print("------------------------------------")
# Her bir gösterge için son performans metriklerinin değerlerini ekrana yazdırma
for indicator, metrics in performance_metrics.items():
    print(f"\nGösterge: {indicator}")
    for metric_name, values in metrics.items():
        if values:  # Değerler listesi boş olmadığını kontrol et
            last_value = values[-1]  # Listenin son değerini al
            print(f"{metric_name}: {last_value}")
        else:
            print(f"{metric_name}: Liste boş")
"""          
# Sonuçların yazdırılması
print("1. Sharpe Oranı Sıralaması:")
for indicator, rank in sharpe_rank.items():
    print(f"{indicator}: {performance_metrics[indicator]['Sharpe Ratio'][-1]}")

print("\n2. Sortino Oranı Sıralaması:")
for indicator, rank in sortino_rank.items():
    print(f"{indicator}: {performance_metrics[indicator]['Sortino Ratio'][-1]}")

print("\n3. Max Drawdown Sıralaması (En Düşük):")
for indicator, rank in max_drawdown_rank.items():
    print(f"{indicator}: {performance_metrics[indicator]['Max Drawdown'][-1]}")

print("\n4. Calmar Oranı Sıralaması:")
for indicator, rank in calmar_rank.items():
    print(f"{indicator}: {performance_metrics[indicator]['Calmar Ratio'][-1]}")

print("\n5. Gain/Loss Oranı Sıralaması:")
for indicator, rank in gain_loss_rank.items():
    print(f"{indicator}: {performance_metrics[indicator]['Gain/Loss Ratio'][-1]}")
print("\nGenel Sıralama (1/Puan'a Göre):")
print("Gösterge\tSharpe Oranı\tSortino Oranı\tMax Drawdown\tCalmar Oranı\tGain/Loss Oranı\tToplam Puan\t1/Puan")
for indicator, score in ranked_indicators:
    total_score = 1 / score
    print(f"{indicator}\t{sharpe_rank[indicator]}\t{sortino_rank[indicator]}\t{max_drawdown_rank[indicator]}\t{calmar_rank[indicator]}\t{gain_loss_rank[indicator]}\t{total_score:.2f}")
toplam_skor=0
print("\nSonuç:")
for rank, (indicator, score) in enumerate(ranked_indicators, 1):
    total_score = 1 / score
    toplam_skor+=total_score
for rank, (indicator, score) in enumerate(ranked_indicators, 1):
    total_score = 1 / score
    print(f"{rank}. {indicator}: Yüzdesel puan:",(total_score/toplam_skor)*100)
