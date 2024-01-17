#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
import sys
import warnings

NUMCH = 10658  # total number of lipid head
DP = int(sys.argv[1])  # number of monomer on each chain
NP = int(sys.argv[2])  # number of polymer chains
XML = sys.argv[3]  # name of xml file
DCD = sys.argv[4]  # name of dcd file
OUTFILE = "FindLoop.dat"  # name of output file
ADSDIS = 2.5  # max adsorption distance
RANGE = 2.0  # neighbor lipid on xy plane
LX = 80
LY = 80
LZ = 40


RANGESQ = RANGE**2
ADSDISSQ = ADSDIS**2
NUMP = NP * DP
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2
tot_train, tot_tail, tot_loops = 0, 0, 0
frame = 0


warnings.filterwarnings("ignore")


@nb.njit
def fold_back(P):
    P[0] = (P[0] + HALFLX) % LX - HALFLX
    P[1] = (P[1] + HALFLY) % LY - HALFLY
    P[2] = (P[2] + HALFLZ) % LZ - HALFLZ


@nb.njit
def find(Pxyz, CHxyz):
    train, tail, loops = 0, 0, 0
    dis = []

    for i in range(
        0, NUMP
    ):  # for each monomer, find the closest lipid head and save the distance
        distancesq = []
        for j in range(0, NUMCH):
            rsq = (Pxyz[i][0] - CHxyz[j][0]) ** 2 + (Pxyz[i][1] - CHxyz[j][1]) ** 2
            if rsq <= RANGESQ:
                distancesq.append(rsq + (Pxyz[i][2] - CHxyz[j][2]) ** 2)
        dis.append(min(distancesq))

    for i in range(0, NP):
        chain = dis[i * DP : (i + 1) * DP]
        binchain = [1 if x < ADSDISSQ else 0 for x in chain]
        train += sum(binchain)
        for j in range(0, DP):
            if binchain[j] == 1:
                tail += j
                break
        for j in range(DP - 1, -1, -1):
            if binchain[j] == 1:
                tail += (DP - 1) - j
                break
        loops = NP * DP - train - tail
    return train, tail, loops


U = mda.Universe(XML, DCD)
P = U.select_atoms("type P")
C = U.select_atoms("type C")
H = U.select_atoms("type H")
CH = C + H

with open(OUTFILE, "w") as f:
    f.write(f" frame \t train \t tail \t loops \n")

for ts in U.trajectory[1:]:
    Pxyz = P.positions
    CHxyz = CH.positions
    for i in range(0, NUMP):
        fold_back(Pxyz[i])
    for i in range(0, NUMCH):
        fold_back(CHxyz[i])

    train, tail, loops = find(Pxyz, CHxyz)
    tot_train += train
    tot_tail += tail
    tot_loops += loops

    frame += 1

    with open(OUTFILE, "a") as f:
        f.write(f"{frame:^7}\t{train:^7}\t{tail:^6}\t{loops:^7}\n")


aver_train = tot_train / frame
aver_tail = tot_tail / frame
aver_loops = tot_loops / frame

with open(OUTFILE, "a") as f:
    f.write(
        f"train_average: {aver_train}\ntail_average: {aver_tail}\nloops_average: {aver_loops}\n\n"
    )
