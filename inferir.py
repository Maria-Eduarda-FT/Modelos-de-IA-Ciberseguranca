"""Carrega um modelo e classifica as N primeiras instâncias do conjunto de teste.
python inferir.py {dt|nb|nn} N   N=0: só carrega o modelo"""
import sys
import time

import joblib
import numpy as np

PASTA = "modelos_ransomware"
ARQUIVOS = {"dt": "arvore_decisao.joblib", "nb": "naive_bayes.joblib", "nn": "rede_neural.keras"}


def inferir(nome, n, pasta=PASTA):
    teste = np.load(f"{pasta}/teste.npz")
    X, y = teste["X"][:n], teste["y"][:n]

    caminho = f"{pasta}/{ARQUIVOS[nome]}"
    if nome == "nn":  # TensorFlow só é importado quando necessário
        from tensorflow import keras
        modelo = keras.models.load_model(caminho)
    else: modelo = joblib.load(caminho)

    if n == 0: print(f"{nome}: modelo carregado (sem predicao)"); return

    if nome != "dt":
        X = joblib.load(f"{pasta}/scaler.joblib").transform(X)

    t0 = time.perf_counter()

    if nome == "nn": 
        pred = (modelo.predict(X, verbose=0) > 0.5).astype(int).ravel()
    else: 
        pred = modelo.predict(X)

    ms = (time.perf_counter() - t0) * 1000
    print(f"{nome}: n={n} acertos={np.mean(pred == y):.4f} predict={ms:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ARQUIVOS:
        sys.exit(__doc__)
    inferir(sys.argv[1], int(sys.argv[2]))
