#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
import sys
import warnings

DP = int(sys.argv[1])  # number of monomer on each chain
NP = int(sys.argv[2])  # number of polymer chains
LX = float(sys.argv[3])
LY = float(sys.argv[4])
LZ = float(sys.argv[5])
XML = sys.argv[6]  # name of xml file
DCD = sys.argv[7]  # name of dcd file
OUTFILE = "FindLoopDistance.dat"  # name of output file
ADSDIS = 1.25  # max adsorption distance
RCUT = 2.5
RANGE = RCUT


RANGESQ = RANGE**2
ADSDISSQ = ADSDIS**2
NUMP = NP * DP
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2
tot_train = []
tot_tail = []
tot_loops = []
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

    for i in range(0, NUMP):  # find the closest lipid head and save the distance
        distancesq = []
        for j in range(0, len(CHxyz)):
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


@nb.jit
def Calc_dat(tot_train, tot_tail, tot_loops):
    aver_train = np.average(tot_train)
    aver_tail = np.average(tot_tail)
    aver_loops = np.average(tot_loops)
    max_train = np.max(tot_train)
    max_tail = np.max(tot_tail)
    max_loops = np.max(tot_loops)
    min_train = np.min(tot_train)
    min_tail = np.min(tot_tail)
    min_loops = np.min(tot_loops)
    return (
        aver_train,
        aver_tail,
        aver_loops,
        max_train,
        max_tail,
        max_loops,
        min_train,
        min_tail,
        min_loops,
    )


U = mda.Universe(XML, DCD)
P = U.select_atoms("type P")
C = U.select_atoms("type C")
H = U.select_atoms("type H")
CH = C + H

with open(OUTFILE, "w") as f:
    f.write(f" frame \t train \t tail \t loops \t rate \n")

for ts in U.trajectory[1:]:
    Pxyz = P.positions
    CHxyz = CH.positions
    for i in range(0, NUMP):
        fold_back(Pxyz[i])
    for i in range(0, len(CHxyz)):
        fold_back(CHxyz[i])

    train, tail, loops = find(Pxyz, CHxyz)
    tot_train.append(train)
    tot_tail.append(tail)
    tot_loops.append(loops)
    frame += 1

    with open(OUTFILE, "a") as f:
        f.write(
            f"{frame:^7}\t{train:^7}\t{tail:^6}\t{loops:^7}\t{(tail+loops)/(DP*NP):^6}\n"
        )

(
    aver_train,
    aver_tail,
    aver_loops,
    max_train,
    max_tail,
    max_loops,
    min_train,
    min_tail,
    min_loops,
) = Calc_dat(tot_train, tot_tail, tot_loops)


with open(OUTFILE, "a") as f:
    f.write(
        f"train_average: {aver_train:^7}\ttail_average: {aver_tail:^8}\tloops_average: {aver_loops:^7}\tDisorption rate: {(aver_loops+aver_tail)/(DP*NP)}\n"
    )
    f.write(
        f"train_maximum: {max_train:^7}\ttail_maximum: {max_tail:^8}\tloops_maximum: {max_loops:^7}\n"
    )
    f.write(
        f"train_minimum: {min_train:^7}\ttail_minimum: {min_tail:^8}\tloops_minimum: {min_loops:^7}\n"
    )
