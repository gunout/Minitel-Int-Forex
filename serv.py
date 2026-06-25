#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pytz
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

US_TIMEZONE = pytz.timezone('America/New_York')
cache = {}
CACHE_DURATION = 30

# ============================================================
# COMMODITIES SYMBOLS
# ============================================================

COMMODITY_DATA = {
    # Métaux précieux
    'GC=F': {'name': 'Or', 'exchange': 'COMEX', 'category': 'Métaux'},
    'SI=F': {'name': 'Argent', 'exchange': 'COMEX', 'category': 'Métaux'},
    'PL=F': {'name': 'Platine', 'exchange': 'NYMEX', 'category': 'Métaux'},
    'PA=F': {'name': 'Palladium', 'exchange': 'NYMEX', 'category': 'Métaux'},

    # Énergie
    'CL=F': {'name': 'Pétrole WTI', 'exchange': 'NYMEX', 'category': 'Énergie'},
    'BZ=F': {'name': 'Pétrole Brent', 'exchange': 'ICE', 'category': 'Énergie'},
    'NG=F': {'name': 'Gaz Naturel', 'exchange': 'NYMEX', 'category': 'Énergie'},
    'RB=F': {'name': 'Essence RBOB', 'exchange': 'NYMEX', 'category': 'Énergie'},
    'HO=F': {'name': 'Fioul', 'exchange': 'NYMEX', 'category': 'Énergie'},

    # Agricoles
    'ZC=F': {'name': 'Maïs', 'exchange': 'CBOT', 'category': 'Agricole'},
    'ZW=F': {'name': 'Blé', 'exchange': 'CBOT', 'category': 'Agricole'},
    'ZS=F': {'name': 'Soja', 'exchange': 'CBOT', 'category': 'Agricole'},
    'ZM=F': {'name': 'Farine de Soja', 'exchange': 'CBOT', 'category': 'Agricole'},
    'ZL=F': {'name': 'Huile de Soja', 'exchange': 'CBOT', 'category': 'Agricole'},

    # Softs
    'KC=F': {'name': 'Café', 'exchange': 'ICE', 'category': 'Softs'},
    'CC=F': {'name': 'Cacao', 'exchange': 'ICE', 'category': 'Softs'},
    'SB=F': {'name': 'Sucre', 'exchange': 'ICE', 'category': 'Softs'},
    'CT=F': {'name': 'Coton', 'exchange': 'ICE', 'category': 'Softs'},

    # Métaux de base
    'HG=F': {'name': 'Cuivre', 'exchange': 'COMEX', 'category': 'Métaux'},

    # Bétail
    'LE=F': {'name': 'Bovins Vivants', 'exchange': 'CME', 'category': 'Bétail'},
    'HE=F': {'name': 'Porcs', 'exchange': 'CME', 'category': 'Bétail'},

    # ETFs
    'DBC': {'name': 'DB Commodity ETF', 'exchange': 'NYSE', 'category': 'ETF'},
    'GSG': {'name': 'S&P GSCI ETF', 'exchange': 'NYSE', 'category': 'ETF'},
}

# ============================================================
# WATCHLIST COMPLÈTE
# ============================================================

WATCHLIST = [
    'GC=F', 'SI=F', 'PL=F', 'PA=F',  # Métaux précieux
    'CL=F', 'BZ=F', 'NG=F',           # Énergie
    'ZC=F', 'ZW=F', 'ZS=F',           # Agricoles
    'KC=F', 'CC=F', 'SB=F',           # Softs
    'HG=F',                            # Métaux de base
    'LE=F', 'HE=F',                   # Bétail
    'DBC', 'GSG'                      # ETFs
]

# ============================================================
# FONCTIONS
# ============================================================

def safe_float(v, default=0.0):
    try:
        if pd.isna(v) or v is None:
            return default
        return float(v)
    except:
        return default

def safe_int(v, default=0):
    try:
        if pd.isna(v) or v is None:
            return default
        return int(v)
    except:
        return default

def get_cached(key):
    if key in cache:
        data, ts = cache[key]
        if (datetime.now() - ts).seconds < CACHE_DURATION:
            return data
    return None

def set_cached(key, data):
    cache[key] = (data, datetime.now())

# ============================================================
# ROUTES
# ============================================================

@app.route('/api/clear-cache')
def clear_cache():
    cache.clear()
    return jsonify({'status': 'ok'})

@app.route('/api/trading/<symbol>')
def get_trading(symbol):
    try:
        cached = get_cached(f"trading_{symbol}")
        if cached:
            return jsonify(cached)

        logger.info(f"Fetching {symbol}")
        ticker = yf.Ticker(symbol)

        hist_test = ticker.history(period='1d')
        if hist_test.empty:
            return jsonify({'error': f'Symbole {symbol} non trouvé'}), 404

        periods = {
            '1d': '1m',
            '5d': '5m',
            '1mo': '15m',
            '3mo': '1h',
            '6mo': '1d',
            '1y': '1d'
        }

        commodity_info = COMMODITY_DATA.get(symbol, {})

        result = {
            'symbol': symbol,
            'name': commodity_info.get('name', symbol),
            'exchange': commodity_info.get('exchange', 'Commodity'),
            'currency': 'USD',
            'data': {}
        }

        for period, interval in periods.items():
            try:
                hist = ticker.history(period=period, interval=interval)
                if hist.empty:
                    continue

                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize('UTC').tz_convert(US_TIMEZONE)
                else:
                    hist.index = hist.index.tz_convert(US_TIMEZONE)

                close = hist['Close'].values
                high = hist['High'].values
                low = hist['Low'].values

                candles = []
                for idx, row in hist.iterrows():
                    candles.append({
                        'time': int(idx.timestamp()),
                        'open': safe_float(row['Open']),
                        'high': safe_float(row['High']),
                        'low': safe_float(row['Low']),
                        'close': safe_float(row['Close']),
                        'volume': safe_int(row['Volume'])
                    })

                if not candles:
                    continue

                result['data'][period] = {
                    'candles': candles,
                    'stats': {
                        'current_price': safe_float(close[-1]),
                        'change': safe_float(close[-1] - close[-2]) if len(close) > 1 else 0,
                        'change_percent': safe_float(((close[-1] - close[-2]) / close[-2] * 100)) if len(close) > 1 and close[-2] != 0 else 0,
                        'high': safe_float(max(high)),
                        'low': safe_float(min(low)),
                        'volume': safe_int(hist['Volume'].sum())
                    }
                }

            except Exception as e:
                logger.error(f"Erreur {period} {symbol}: {e}")
                continue

        if not result['data']:
            return jsonify({'error': f'Aucune donnée pour {symbol}'}), 404

        set_cached(f"trading_{symbol}", result)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights-advanced/<symbol>')
def get_insights(symbol):
    try:
        cached = get_cached(f"insights_{symbol}")
        if cached:
            return jsonify(cached)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')

        if hist.empty or len(hist) < 30:
            return jsonify({'error': 'Pas assez de données'})

        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        current = safe_float(close[-1])

        returns = np.diff(close) / close[:-1]
        vol = safe_float(np.std(returns) * np.sqrt(252) * 100) if len(returns) > 0 else 0

        support = safe_float(np.percentile(low[-30:], 20)) if len(low) >= 30 else safe_float(min(low))
        resistance = safe_float(np.percentile(high[-30:], 80)) if len(high) >= 30 else safe_float(max(high))

        momentum = safe_float((close[-1] - close[-20]) / close[-20] * 100) if len(close) >= 20 else 0

        try:
            x = np.arange(len(close)).reshape(-1, 1)
            y = close.reshape(-1, 1)
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            model.fit(x, y)
            future = np.arange(len(close), len(close) + 5).reshape(-1, 1)
            predictions = model.predict(future).flatten()
            predictions = [safe_float(p) for p in predictions]
        except:
            predictions = [current] * 5

        rsi = 50
        if len(returns) >= 14:
            gains = [r for r in returns[-14:] if r > 0]
            losses = [abs(r) for r in returns[-14:] if r < 0]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            if avg_loss > 0:
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        signals = []
        if rsi > 70:
            signals.append({'type': 'sell', 'indicator': 'RSI', 'value': f'{rsi:.1f}', 'message': 'Surachat'})
        elif rsi < 30:
            signals.append({'type': 'buy', 'indicator': 'RSI', 'value': f'{rsi:.1f}', 'message': 'Survente'})

        if current > 0 and support > 0 and (current - support) / current < 0.015:
            signals.append({'type': 'buy', 'indicator': 'Support', 'value': f'{support:.2f}', 'message': 'Proche support'})

        if current > 0 and resistance > 0 and (resistance - current) / current < 0.015:
            signals.append({'type': 'sell', 'indicator': 'Résistance', 'value': f'{resistance:.2f}', 'message': 'Proche résistance'})

        if signals:
            buy_count = sum(1 for s in signals if s['type'] == 'buy')
            sell_count = sum(1 for s in signals if s['type'] == 'sell')
            if buy_count > sell_count:
                rec = 'ACHAT'
                conf = min(90, 50 + buy_count * 15)
            elif sell_count > buy_count:
                rec = 'VENTE'
                conf = min(90, 50 + sell_count * 15)
            else:
                rec = 'NEUTRE'
                conf = 50
        else:
            rec = 'NEUTRE'
            conf = 50

        result = {
            'current_price': current,
            'volatility': vol,
            'momentum': momentum,
            'supports': [support],
            'resistances': [resistance],
            'predictions': predictions,
            'signals': signals,
            'recommendation': rec,
            'confidence': conf,
            'stop_loss': safe_float(current * 0.975),
            'take_profit': safe_float(current * 1.05),
            'rsi': rsi,
            'macd': 0
        }

        set_cached(f"insights_{symbol}", result)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur insights {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist')
def get_watchlist():
    try:
        results = []
        for symbol in WATCHLIST:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')

                current = safe_float(info.get('regularMarketPrice', 0))
                if current == 0 and not hist.empty:
                    current = safe_float(hist['Close'].iloc[-1])

                prev = safe_float(info.get('regularMarketPreviousClose', 0))
                if prev == 0 and len(hist) > 1:
                    prev = safe_float(hist['Close'].iloc[-2])

                change_pct = ((current - prev) / prev * 100) if prev else 0

                commodity_info = COMMODITY_DATA.get(symbol, {})

                results.append({
                    'symbol': symbol,
                    'name': commodity_info.get('name', symbol),
                    'price': current,
                    'changePercent': change_pct,
                    'change': current - prev,
                    'currency': 'USD'
                })
            except Exception as e:
                logger.warning(f"Erreur watchlist {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-performers')
def get_top_performers():
    try:
        symbols = ['GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'CC=F', 'SB=F', 'HG=F', 'DBC', 'GSG']
        performers = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')

                current = safe_float(info.get('regularMarketPrice', 0))
                if current == 0 and not hist.empty:
                    current = safe_float(hist['Close'].iloc[-1])

                prev = safe_float(info.get('regularMarketPreviousClose', 0))
                if prev == 0 and len(hist) > 1:
                    prev = safe_float(hist['Close'].iloc[-2])

                change_pct = ((current - prev) / prev * 100) if prev else 0

                commodity_info = COMMODITY_DATA.get(symbol, {})

                performers.append({
                    'symbol': symbol,
                    'name': commodity_info.get('name', symbol),
                    'price': current,
                    'changePercent': change_pct,
                    'currency': 'USD'
                })
            except:
                continue

        performers.sort(key=lambda x: x['changePercent'], reverse=True)
        return jsonify(performers[:10])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-status')
def market_status():
    now = datetime.now(US_TIMEZONE)
    is_open = now.weekday() < 5 and 9 <= now.hour <= 16
    return jsonify({
        'status': 'open' if is_open else 'closed',
        'label': 'Ouvert' if is_open else 'Fermé',
        'icon': '🟢' if is_open else '🔴',
        'time': now.strftime('%H:%M:%S')
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)

    print("=" * 60)
    print("🛢️ COMMODITIES TRADER")
    print("=" * 60)
    print("🌐 http://localhost:5003")
    print("=" * 60)
    print("📈 Commodités disponibles:")
    for sym, info in COMMODITY_DATA.items():
        print(f"   {sym} - {info['name']} ({info['exchange']})")
    print("=" * 60)
    print("📋 Watchlist:")
    for sym in WATCHLIST:
        print(f"   {sym}")
    print("=" * 60)
    print("💡 Pour vider le cache: http://localhost:5003/api/clear-cache")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5003, debug=True)
