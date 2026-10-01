"""Gera DatasetPersonalizado.csv e Class_ransomware.csv (0=Benign, 1=Ransomware)"""
import pandas as pd

ORIGINAL = "Obfuscated-MalMem2022.csv"
SAIDA_X = "DatasetPersonalizado.csv"
SAIDA_Y = "Class_ransomware.csv"
LIMITE_BENIGN = 10020
REMOVER = [  # pouca variabilidade
    "pslist.nproc", "pslist.nppid", "pslist.nprocs64bit", "psxview.not_in_eprocess_pool",
    "psxview.not_in_ethread_pool", "psxview.not_in_session", "psxview.not_in_deskthrd",
    "psxview.not_in_eprocess_pool_false_avg", "modules.nmodules", "svcscan.fs_drivers",
    "svcscan.nactive", "callbacks.ncallbacks", "callbacks.nanonymous", "callbacks.ngeneric",
]


def preparar(original=ORIGINAL, saida_x=SAIDA_X, saida_y=SAIDA_Y):
    df = pd.read_csv(original)
    benigno = df["Category"].str.contains("Benign", na=False)
    ransom = df["Category"].str.contains("Ransomware", na=False)
    df = pd.concat([df[benigno].head(LIMITE_BENIGN), df[ransom]], ignore_index=True)

    y = (~df["Category"].str.contains("Benign", na=False)).astype(int)
    X = df.drop(columns=["Category", "Class"] + REMOVER, errors="ignore")

    X.to_csv(saida_x, index=False)
    y.to_frame("Class").to_csv(saida_y, index=False)
    print(f"{saida_x}: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"{saida_y}: Benign={(y == 0).sum()}  Ransomware={(y == 1).sum()}")


if __name__ == "__main__":
    preparar()
