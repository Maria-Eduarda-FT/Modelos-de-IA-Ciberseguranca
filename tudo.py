"""Pipeline completo: dataset -> treino -> inferência -> straces
  python tudo.py                 # dataset + treino + straces (tudo)
  python tudo.py dataset         # gera DatasetPersonalizado.csv e Class_ransomware.csv
  python tudo.py treinar         # treina e salva os modelos
  python tudo.py inferir nn 100  # modelo {dt|nb|nn} e nº de instâncias
  python tudo.py straces [--cargas 0 1 100 1000] [--rep 3] [--completo] [--retreinar]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

PASTA = "modelos_ransomware"
SEED = 42
ARQUIVOS = {"dt": "arvore_decisao.joblib", "nb": "naive_bayes.joblib", "nn": "rede_neural.keras"}
ORIGINAL, SAIDA_X, SAIDA_Y = "Obfuscated-MalMem2022.csv", "DatasetPersonalizado.csv", "Class_ransomware.csv"
LIMITE_BENIGN = 10020
REMOVER = [  # pouca variabilidade
    "pslist.nproc", "pslist.nppid", "pslist.nprocs64bit", "psxview.not_in_eprocess_pool",
    "psxview.not_in_ethread_pool", "psxview.not_in_session", "psxview.not_in_deskthrd",
    "psxview.not_in_eprocess_pool_false_avg", "modules.nmodules", "svcscan.fs_drivers",
    "svcscan.nactive", "callbacks.ncallbacks", "callbacks.nanonymous", "callbacks.ngeneric",
]
CLONES = {"clone", "clone3", "fork", "vfork"}


# ---------------------------------------------------------------- dataset
def preparar_dataset():
    import pandas as pd
    df = pd.read_csv(ORIGINAL)
    benigno = df["Category"].str.contains("Benign", na=False)
    ransom = df["Category"].str.contains("Ransomware", na=False)
    df = pd.concat([df[benigno].head(LIMITE_BENIGN), df[ransom]], ignore_index=True)
    y = (~df["Category"].str.contains("Benign", na=False)).astype(int)
    X = df.drop(columns=["Category", "Class"] + REMOVER, errors="ignore")
    X.to_csv(SAIDA_X, index=False)
    y.to_frame("Class").to_csv(SAIDA_Y, index=False)
    print(f"{SAIDA_X}: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"{SAIDA_Y}: Benign={(y == 0).sum()}  Ransomware={(y == 1).sum()}")


# ---------------------------------------------------------------- treino
def treinar():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X = pd.read_csv(SAIDA_X).select_dtypes(include=[np.number])
    X = X.fillna(X.mean()).values
    y = pd.read_csv(SAIDA_Y)["Class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"Treino/teste: {len(X_tr)}/{len(X_te)} | features: {X_tr.shape[1]}")

    # Árvore (dados sem normalizar)
    grade = {"max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 5, 10, 20],
             "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", "log2", None]}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED), grade, cv=5, scoring="f1", n_jobs=-1)
    dt = gs.fit(X_tr, y_tr).best_estimator_
    print(f"Árvore: {gs.best_params_} | F1 CV = {gs.best_score_:.4f}")

    # Naive Bayes 
    candidatos = [GaussianNB(), GaussianNB(var_smoothing=1e-8)]
    scores = [cross_val_score(m, Xs_tr, y_tr, cv=5, scoring="f1").mean() for m in candidatos]
    nb = candidatos[int(np.argmax(scores))].fit(Xs_tr, y_tr)
    print(f"Naive Bayes: var_smoothing={nb.var_smoothing} | F1 CV = {max(scores):.4f}")

    # Rede neural 
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.keras.utils.set_random_seed(SEED)
    init = "random_uniform"
    nn = Sequential([
        layers.Input(shape=(Xs_tr.shape[1],)),
        layers.Dense(128, activation="relu", kernel_initializer=init), layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_initializer=init), layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer=init), layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    nn.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["binary_accuracy"])
    nn.fit(Xs_tr, y_tr, batch_size=32, epochs=100, validation_split=0.2, verbose=2, callbacks=[
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001)])

    os.makedirs(PASTA, exist_ok=True)
    joblib.dump(dt, f"{PASTA}/{ARQUIVOS['dt']}")
    joblib.dump(nb, f"{PASTA}/{ARQUIVOS['nb']}")
    nn.save(f"{PASTA}/{ARQUIVOS['nn']}")
    joblib.dump(scaler, f"{PASTA}/scaler.joblib")
    np.savez(f"{PASTA}/teste.npz", X=X_te, y=y_te)

    previsoes = {"Árvore": dt.predict(X_te), "Naive Bayes": nb.predict(Xs_te),
                 "Rede Neural": (nn.predict(Xs_te, verbose=0) > 0.5).astype(int).ravel()}
    print("\nResultado no conjunto de teste:")
    for nome, p in previsoes.items():
        print(f"  {nome:<12} acurácia={accuracy_score(y_te, p):.4f}  F1={f1_score(y_te, p):.4f}")
    print(f"Modelos salvos em {PASTA}/")


# ---------------------------------------------------------------- inferência
def inferir(nome, n):
    import joblib  
    import numpy as np
    teste = np.load(f"{PASTA}/teste.npz")
    X, y = teste["X"][:n], teste["y"][:n]
    caminho = f"{PASTA}/{ARQUIVOS[nome]}"
    if nome == "nn": # TensorFlow so entra para a rede neural
        from tensorflow import keras
        modelo = keras.models.load_model(caminho)
    else:
        modelo = joblib.load(caminho)
    if n == 0:
        print(f"{nome}: modelo carregado (sem predição)")
        return
    if nome != "dt":
        X = joblib.load(f"{PASTA}/scaler.joblib").transform(X)
    t0 = time.perf_counter()
    if nome == "nn":
        pred = (modelo.predict(X, verbose=0) > 0.5).astype(int).ravel()
    else:
        pred = modelo.predict(X)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{nome}: n={n} acertos={np.mean(pred == y):.4f} predict={ms:.2f} ms")


# ---------------------------------------------------------------- straces
def garantir_modelos(retreinar=False):
    if not (os.path.exists(SAIDA_X) and os.path.exists(SAIDA_Y)):
        preparar_dataset()
    prontos = all(os.path.exists(f"{PASTA}/{a}") for a in ARQUIVOS.values())
    if retreinar or not prontos or not os.path.exists(f"{PASTA}/teste.npz"):
        treinar()


def resumir(arquivo):
    """le a tabela do `strace -c` e soma syscalls, erros, tipos distintos, clones e execve"""
    r = dict(syscalls=0, erros=0, distintas=0, clones=0, execve=0, tempo_sys_s=0.0)
    for linha in open(arquivo, errors="ignore"):
        p = linha.replace(",", ".").split()
        if len(p) < 5 or not p[0].replace(".", "", 1).isdigit() or p[-1] == "total":
            continue
        calls = int(p[3])
        r["syscalls"] += calls
        r["erros"] += int(p[4]) if len(p) == 6 else 0  # sem erros, a coluna some
        r["distintas"] += 1
        r["tempo_sys_s"] += float(p[1])
        r["clones"] += calls if p[-1] in CLONES else 0
        r["execve"] += calls if p[-1] == "execve" else 0
    return r


def straces(cargas, rep, saida, completo, retreinar):
    import pandas as pd
    if not shutil.which("strace"):
        sys.exit("strace não encontrado: sudo apt install strace")
    garantir_modelos(retreinar)
    os.makedirs(saida, exist_ok=True)
    linhas = []
    for r in range(1, rep + 1):
        for modelo in ARQUIVOS:
            for n in cargas:
                base = f"{saida}/{modelo}_{n}_r{r}"
                cmd = [sys.executable, os.path.abspath(__file__), "inferir", modelo, str(n)]
                print(f"[rep {r}] {modelo} carga={n}", flush=True)
                subprocess.run(["strace", "-f", "-c", "-o", f"{base}.txt"] + cmd,
                               check=True, stdout=subprocess.DEVNULL)
                if completo:
                    subprocess.run(["strace", "-f", "-tt", "-T", "-o", f"{base}.log"] + cmd,
                                   check=True, stdout=subprocess.DEVNULL)
                linhas.append(dict(modelo=modelo, carga=n, rep=r, **resumir(f"{base}.txt")))
    df = pd.DataFrame(linhas)
    df.to_csv(f"{saida}/resumo.csv", index=False)
    print("\nMédia das repetições:")
    print(df.drop(columns="rep").groupby(["modelo", "carga"]).mean().round(2).to_string())
    print(f"\nResumo salvo em {saida}/resumo.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", nargs="?", default="tudo",
                    choices=["tudo", "dataset", "treinar", "inferir", "straces"])
    ap.add_argument("args", nargs="*", help="para 'inferir': modelo (dt|nb|nn) e N")
    ap.add_argument("--cargas", type=int, nargs="+", default=[0, 1, 100, 1000])
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--saida", default="straces")
    ap.add_argument("--completo", action="store_true", help="salva também o trace completo (-f -tt -T)")
    ap.add_argument("--retreinar", action="store_true")
    a = ap.parse_args()

    if a.etapa == "dataset":
        preparar_dataset()
    elif a.etapa == "treinar":
        treinar()
    elif a.etapa == "inferir":
        if len(a.args) != 2 or a.args[0] not in ARQUIVOS:
            sys.exit("uso: python tudo.py inferir {dt|nb|nn} N")
        inferir(a.args[0], int(a.args[1]))
    else:  # straces já prepara dataset e modelos se faltarem
        straces(a.cargas, a.rep, a.saida, a.completo, a.retreinar)


if __name__ == "__main__":
    main()
