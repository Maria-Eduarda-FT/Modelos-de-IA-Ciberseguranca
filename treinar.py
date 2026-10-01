"""salva tudo em modelos_ransomware/"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PASTA = "modelos_ransomware"
SEED = 42

def carregar(dados="DatasetPersonalizado.csv", classes="Class_ransomware.csv"):
    X = pd.read_csv(dados).select_dtypes(include=[np.number])
    X = X.fillna(X.mean()).values
    y = pd.read_csv(classes)["Class"].values
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

def treinar_arvore(X, y):
    grade = {"max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 5, 10, 20],
             "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", "log2", None]}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED), grade, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X, y)
    print(f"Árvore: {gs.best_params_} | F1 CV = {gs.best_score_:.4f}")
    return gs.best_estimator_

def treinar_naive_bayes(X, y):
    candidatos = [GaussianNB(), GaussianNB(var_smoothing=1e-8)]
    scores = [cross_val_score(m, X, y, cv=5, scoring="f1").mean() for m in candidatos]
    melhor = candidatos[int(np.argmax(scores))]
    print(f"Naive Bayes: var_smoothing={melhor.var_smoothing} | F1 CV = {max(scores):.4f}")
    return melhor.fit(X, y)

def treinar_rede(X, y):
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    tf.keras.utils.set_random_seed(SEED)
    init = "random_uniform"
    modelo = Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation="relu", kernel_initializer=init), layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_initializer=init), layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer=init), layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy",
                   metrics=["binary_accuracy"])
    modelo.fit(X, y, batch_size=32, epochs=100, validation_split=0.2, verbose=2, callbacks=[
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001)])
    return modelo

def treinar(pasta=PASTA):
    X_tr, X_te, y_tr, y_te = carregar()
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"Treino/teste: {len(X_tr)}/{len(X_te)} | features: {X_tr.shape[1]}")

    dt = treinar_arvore(X_tr, y_tr)           # árvore usa dados sem normalizar
    nb = treinar_naive_bayes(Xs_tr, y_tr)     # NB e rede usam dados normalizados
    nn = treinar_rede(Xs_tr, y_tr)

    os.makedirs(pasta, exist_ok=True)
    joblib.dump(dt, f"{pasta}/arvore_decisao.joblib")
    joblib.dump(nb, f"{pasta}/naive_bayes.joblib")
    nn.save(f"{pasta}/rede_neural.keras")
    joblib.dump(scaler, f"{pasta}/scaler.joblib")
    np.savez(f"{pasta}/teste.npz", X=X_te, y=y_te)  # conjunto de teste (sem normalizar)

    previsoes = {"Árvore": dt.predict(X_te), "Naive Bayes": nb.predict(Xs_te),
                 "Rede Neural": (nn.predict(Xs_te, verbose=0) > 0.5).astype(int).ravel()}
    print("\nResultado no conjunto de teste:")
    for nome, p in previsoes.items():
        print(f"  {nome:<12} acurácia={accuracy_score(y_te, p):.4f}  F1={f1_score(y_te, p):.4f}")
    print(f"Modelos salvos em {pasta}/")

if __name__ == "__main__":
    treinar()
