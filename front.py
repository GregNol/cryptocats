import apimoex
import streamlit as st
import pandas as pd
import numpy as np
import requests
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sympy import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers_start = 'YNDX VKCO TCSG POLY OZON SBER'.split()
st.title('Аналитика рынка и создание инвестиционного портфеля')


class Ticker:
    def __init__(self, tick, delta, risk, data, deltad):
        self.tick = tick
        self.delta = delta
        self.risk = risk
        self.prices = data
        self.deltad = deltad


ticks = {}


def load_historical_data(ticker):
    global ticks
    session1 = requests.Session()
    historiscal_data = apimoex.get_board_history(session=session1, security=ticker, start='2022-03-15',
                                                 end='2023-03-15',
                                                 board='TQBR')
    data = pd.DataFrame(historiscal_data)
    data = data.iloc[:-1, :]
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'])
    data.set_index('TRADEDATE', inplace=True)
    data.dropna(subset=['CLOSE'])
    data = data[data.CLOSE > 0]
    data = data.reindex(index=data.index[::-1])
    data['deltad'] = data['CLOSE'].rolling(window=2, center=False).apply(lambda x: x[1] / x[0] - 1)
    data = data[data.deltad > 0]
    data = data.reindex(index=data.index[::-1])
    print(data)
    delta = data['deltad']
    delta = sum(delta) / len(delta)
    risk = sum([abs(delta - i) for i in data['deltad']]) / len(data['deltad'])

    ticks[ticker] = Ticker(ticker, delta, risk, list(data['CLOSE']), list(data['deltad']))
    print(list(data['deltad']))
    return data


tickers = st.text_input('Введите тикер вашего инструмента: ', value=' '.join(tickers_start))
# data_load_state = st.text('Загружаем базовую информацию...')
st.subheader('Список котировок для анализа')
st.text(' '.join(tickers))
st.subheader('Котировки цен инсрументов')
for ticker in tickers.split():
    history = load_historical_data(ticker)
    st.subheader(ticker)
    st.line_chart(history['CLOSE'])
D = []
mdx = []
for i in ticks.keys():
    mdx.append(len(ticks[i].deltad))
mdx = min(mdx)
for i in ticks.keys():
    D.append(ticks[i].deltad[:mdx - 1] * 100)
    print(len(ticks[i].deltad))
print(D)
D = np.array(D, np.float64)
print(D)
m, n = D.shape
# history_load_state.text('Информация... Загружена!')
d = np.zeros([m, 1])  # столбец для средней доходности
for i in range(len(ticks.keys())):
    d[i, 0] = ticks[list(ticks.keys())[i]].delta * 100
    print(d[i, 0])
abc = pd.DataFrame([tickers.split(), [i[0] for i in d]])
st.subheader('Ожидаемая доходность, %')
st.dataframe(abc)
print("Средняя доходность акций 1-6 : \n %s" % d)
CV = np.cov(D)
abc = pd.DataFrame(CV)
st.subheader('Ковариационная матрица')
# abc = abc.reindex(index=tickers.split())
st.dataframe(abc)
print("Ковариационная матрица  CV: \n %s" % CV)
max_risk = st.text_input('Какому максимальному риску вы готовы себя подвергнуть(%): ', value=10)
b1, b2, b3, b4, b5, b6 = [None for _ in range(6)]
max_win = 0
for a1 in range(0, 76, 5):
    for a2 in range(0, 101 - a1 - 4 * 5, 5):
        for a3 in range(0, 101 - a1 - a2 - 3 * 5, 5):
            for a4 in range(0, 101 - a1 - a2 - a3 - 2 * 5, 5):
                for a5 in range(0, 101 - a1 - a2 - a3 - a4 - 5, 5):
                    for a6 in range(0, 101 - a1 - a2 - a3 -a4 -a5, 5):
                        if a1 + a2 + a3 + a4 + a5 + a6 != 100:
                            continue
                        win = (a1 * d[0] + a2 * d[1] + a3 * d[2] + a4 * d[3] + a5 * d[4] + a6 * d[5]) / 100
                        lose = round(np.sqrt(np.dot(np.dot([a1, a2, a3, a4, a5, a6], CV), [a1, a2, a3, a4, a5, a6])), 2)
                        if lose <= float(max_risk) and win > max_win:
                            b1, b2, b3, b4, b5, b6 = a1, a2, a3, a4, a5, a6
                            max_win = win
                            print(win, lose)
                            print(b1, b2, b3, b4, b5, b6)
                            # print()
fig1, ax1 = plt.subplots()
ax1.pie([b1, b2, b3, b4, b5, b6], labels=tickers.split())
plt.show()
st.write(pd.DataFrame({'тикер': tickers.split(), 'процент': [b1, b2, b3, b4, b5, b6]}))
b_max = max([b1, b2, b3, b4, b5, b6])
t_max = tickers.split()[[b1, b2, b3, b4, b5, b6].index(b_max)]
st.text(f'Наибольшую важность для инвестора имеют акции компании {t_max}')
