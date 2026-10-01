""" Mede as syscalls da INFERÊNCIA de cada modelo em cada carga, com strace
python rodar_straces.py [--cargas 0 1 100 1000] [--rep 3] [--completo] [--retreinar]
Saída: straces/<modelo>_<carga>_r<rep>.txt  +  straces/resumo.csv"""

import argparse
import os
import shutil
import subprocess
import sys
import pandas as pd
import dataset
import treinar
from inferir import ARQUIVOS, PASTA

CLONES = {"clone", "clone3", "fork", "vfork"}

def garantir_modelos(retreinar):
    if not (os.path.exists(dataset.SAIDA_X) and os.path.exists(dataset.SAIDA_Y)):
        dataset.preparar()
    prontos = all(os.path.exists(f"{PASTA}/{a}") for a in ARQUIVOS.values())
    if retreinar or not prontos or not os.path.exists(f"{PASTA}/teste.npz"):
        treinar.treinar()

def resumir(arquivo):
    """Le a tabela do `strace -c` e soma syscalls, erros, tipos distintos, clones e execve"""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cargas", type=int, nargs="+", default=[0, 1, 100, 1000])
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--saida", default="straces")
    ap.add_argument("--completo", action="store_true", help="salva também o trace completo (-f -tt -T)")
    ap.add_argument("--retreinar", action="store_true")
    a = ap.parse_args()

    if not shutil.which("strace"):
        sys.exit("strace não encontrado: sudo apt install strace")
    garantir_modelos(a.retreinar)
    os.makedirs(a.saida, exist_ok=True)

    linhas = []
    for rep in range(1, a.rep + 1):
        for modelo in ARQUIVOS:
            for n in a.cargas:
                base = f"{a.saida}/{modelo}_{n}_r{rep}"
                cmd = [sys.executable, "inferir.py", modelo, str(n)] # manda inferir os modelos
                print(f"[rep {rep}] {modelo} carga={n}", flush=True)
                subprocess.run(["strace", "-f", "-c", "-o", f"{base}.txt"] + cmd,
                               check=True, stdout=subprocess.DEVNULL) # salva o resumo das syscalls
                if a.completo:
                    subprocess.run(["strace", "-f", "-tt", "-T", "-o", f"{base}.log"] + cmd,
                                   check=True, stdout=subprocess.DEVNULL)
                linhas.append(dict(modelo=modelo, carga=n, rep=rep, **resumir(f"{base}.txt")))

    df = pd.DataFrame(linhas)
    df.to_csv(f"{a.saida}/resumo.csv", index=False)
    media = df.drop(columns="rep").groupby(["modelo", "carga"]).mean().round(2)
    print("\nMedia das repeticoes:")
    print(media.to_string())
    print(f"\nResumo salvo em {a.saida}/resumo.csv")

if __name__ == "__main__":
    main()
