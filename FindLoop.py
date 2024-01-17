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
EPSPC = float(sys.argv[6])
EPSPH = float(sys.argv[7])
XML = sys.argv[8]  # name of xml file
DCD = sys.argv[9]  # name of dcd file
OUTFILE = "FindLoopEnergy.dat"  # name of output file
ADSDIS = 2.5  # max adsorption distance
RCUT = 2.5


ADSDISSQ = ADSDIS**2
RCUTSQ = RCUT**2
NUMP = NP * DP
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2
tot_train, tot_tail, tot_loops = 0, 0, 0
frame = 0


warnings.filterwarnings("ignore")


@nb.njit
def fold_back(xyz):
    for i in range(0, len(xyz)):
        xyz[i][0] = (xyz[i][0] + HALFLX) % LX - HALFLX
        xyz[i][1] = (xyz[i][1] + HALFLY) % LY - HALFLY
        xyz[i][2] = (xyz[i][2] + HALFLZ) % LZ - HALFLZ


@nb.njit
def find(Pxyz, Cxyz, Hxyz):
    train, tail, loops = 0, 0, 0
    P_potential = np.zeros(len(Pxyz))

    for i in range(0, NUMP):
        PCdistancesq = np.zeros(len(Cxyz))
        PHdistancesq = np.zeros(len(Hxyz))
        PC_potential = 0
        PH_potential = 0
        for j in range(0, len(Cxyz)):
            rsq = (
                (Pxyz[i][0] - Cxyz[j][0]) ** 2
                + (Pxyz[i][1] - Cxyz[j][1]) ** 2
                + (Pxyz[i][2] - Cxyz[j][2]) ** 2
            )
            PCdistancesq[j] = rsq
            if rsq < RCUTSQ:
                s2 = 1 / rsq
                s6 = s2 * s2 * s2
                PC_potential += EPSPC * (s6 * (s6 - 1.0) - RCUT)

        for j in range(0, len(Hxyz)):
            rsq = (
                (Pxyz[i][0] - Hxyz[j][0]) ** 2
                + (Pxyz[i][1] - Hxyz[j][1]) ** 2
                + (Pxyz[i][2] - Hxyz[j][2]) ** 2
            )
            PHdistancesq[j] = rsq
            if rsq < RCUTSQ:
                s2 = 1 / rsq
                s6 = s2 * s2 * s2
                PH_potential += EPSPH * (s6 * (s6 - 1.0) - RCUT)
        P_potential[i] = PC_potential + PH_potential

    for i in range(0, NP):
        chain = P_potential[i * DP : (i + 1) * DP]
        binchain = [1 if x < -2.0 else 0 for x in chain]
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

with open(OUTFILE, "w") as f:
    f.write(f" frame \t train \t tail \t loops \n")

for ts in U.trajectory[1:]:
    Pxyz = P.positions
    Cxyz = C.positions
    Hxyz = H.positions
    fold_back(Pxyz)
    fold_back(Cxyz)
    fold_back(Hxyz)

    train, tail, loops = find(Pxyz, Cxyz, Hxyz)
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
