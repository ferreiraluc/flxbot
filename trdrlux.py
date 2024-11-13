import ccxt
import pandas as pd
import numpy as np
import time
import pytz
import os
from datetime import datetime, timedelta
from twilio.rest import Client
from sklearn.linear_model import LinearRegression

# Configuração do Twilio
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_number = 'whatsapp:+14155238886'
my_whatsapp_number = 'whatsapp:+554599532052'
client = Client(account_sid, auth_token)

# Configuração da exchange e do ativo
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'
last_summary_time = datetime.now() - timedelta(minutes=30)  # Garante envio no início

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1440)  # Pega dados de 24 horas (1440 minutos)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['MA20'] + (df['stddev'] * 2)
    df['lower_band'] = df['MA20'] - (df['stddev'] * 2)
    df['SMA9'] = df['close'].rolling(window=9).mean()
    df['SMA21'] = df['close'].rolling(window=21).mean()
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['DM+'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['DM-'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['DI+'] = 100 * (df['DM+'] / df['TR']).ewm(span=14).mean()
    df['DI-'] = 100 * (df['DM-'] / df['TR']).ewm(span=14).mean()
    df['DX'] = 100 * abs((df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-']))
    df['ADX'] = df['DX'].ewm(span=14).mean()
    return df

def send_whatsapp_alert(message):
    client.messages.create(
        from_=twilio_number,
        body=message,
        to=my_whatsapp_number
    )
    print(f"Alerta enviado para o WhatsApp: {message}")

def generate_half_hour_summary(df):
    last_row = df.iloc[-1]
    last_price = last_row['close']
    half_hour_mean = df['close'].tail(30).mean()
    rsi = last_row['RSI']
    adx = last_row['ADX']
    upper_band = last_row['upper_band']
    lower_band = last_row['lower_band']
    
    # Variação percentual de 1 minuto
    percent_change_1m = ((last_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100

    # Variação percentual de 24 horas
    if len(df) >= 1440:
        price_24h_ago = df['close'].iloc[-1440]
        percent_change_24h = ((last_price - price_24h_ago) / price_24h_ago) * 100
        percent_change_24h_text = f"{percent_change_24h:.2f}%"
    else:
        percent_change_24h_text = "Dados insuficientes"

    # Monta a mensagem de resumo
    summary_message = (
        f"Resumo BTC/USDT:\n"
        f"Preço Atual: ${last_price:.2f}\n"
        f"Média (últimos 30 min): ${half_hour_mean:.2f}\n"
        f"RSI: {rsi:.2f}\n"
        f"ADX: {adx:.2f}\n"
        f"Banda Superior: ${upper_band:.2f}\n"
        f"Banda Inferior: ${lower_band:.2f}\n"
        f"Variação 1 min: {percent_change_1m:.2f}%\n"
        f"Variação 24h: {percent_change_24h_text}\n"
    )
    
    send_whatsapp_alert(summary_message)

def determine_trade_signal(df):
    last_row = df.iloc[-1]
    if last_row['close'] <= last_row['lower_band'] and last_row['RSI'] < 30:
        return "LONG"  # Compra, preço tocando banda inferior e RSI baixo (sobrevendido)
    elif last_row['close'] >= last_row['upper_band'] and last_row['RSI'] > 70:
        return "SHORT"  # Venda, preço tocando banda superior e RSI alto (sobrecomprado)
    return None

def predict_price_trend(df):
    # Previsão de tendência de curto prazo usando os últimos 5 minutos
    recent_closes = df['close'].tail(5).values
    X = np.array(range(len(recent_closes))).reshape(-1, 1)
    y = recent_closes
    model = LinearRegression()
    model.fit(X, y)
    trend = "UP" if model.coef_[0] > 0 else "DOWN"
    return trend

def main():
    global last_summary_time
    while True:
        data = fetch_data()
        data = calculate_indicators(data)
        
        # Verifica se é hora de enviar o resumo
        current_time = datetime.now()
        if (current_time - last_summary_time).total_seconds() >= 300:  # 5 minutos
            generate_half_hour_summary(data)
            last_summary_time = current_time  # Atualiza o tempo do último resumo enviado

        # Verifica sinais de trade
        signal = determine_trade_signal(data)
        if signal:
            message = f"Sinal de {signal} detectado para BTC/USDT."
            send_whatsapp_alert(message)
        
        # Previsão de tendência com IA usando últimos 5 minutos
        trend = predict_price_trend(data)
        print(f"Tendência prevista: {trend} = {fetch_data().iloc[-1]['close']:.2f}")

        time.sleep(10)  # Ajuste para atualizar a cada 10 segundos

# Executa o robô
main()
