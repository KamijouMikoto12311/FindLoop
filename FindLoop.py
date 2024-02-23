#!/opt/homebrew/bin/
import xml.etree.ElementTree as et
import MDAnalysis as mda
import numba as nb
import numpy as np
import sys
import os
import re
import warnings

DP = int(sys.argv[1])  # number of monomer on each chain
NP = int(sys.argv[2])  # number of polymer chains
OUTFILE = "FindLoopDistance.dat"  # name of output file
DCDNAME = "traj.dcd"  # name of dcd file
TIMESTEP = 0.01
ADSDIS = 1.5  # max adsorption distance

ADSDISSQ = ADSDIS**2
NUMP = NP * DP
tot_train = []
tot_tail = []
tot_loops = []
frame = 0


warnings.filterwarnings("ignore")


@nb.njit
def fold_back(P, size):
    for i in range(len(P)):
        for dim in range(3):
            P[i][dim] = (P[i][dim] + 0.5 * size[dim]) % size[dim] - 0.5 * size[dim]


@nb.njit
def apply_min_img(r, size):
    for dim in range(3):
        if r[dim] < -0.5 * size[dim]:
            r[dim] += size[dim]
        if r[dim] > 0.5 * size[dim]:
            r[dim] -= size[dim]


@nb.njit
def find(Pxyz, CHxyz, box_size):
    train, tail, loops = 0, 0, 0
    dis = []

    for i in range(0, len(Pxyz)):  # find the closest lipid head and save the distance
        distancesq = []

        for j in range(0, len(CHxyz)):
            r = Pxyz[i] - CHxyz[j]
            apply_min_img(r, box_size)
            ijdistancesq = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            distancesq.append(ijdistancesq)

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


current_dir = os.getcwd()
dirs = [
    x for x in os.listdir(current_dir) if (os.path.isdir(x) and re.match(r"[sr]\d+", x))
]
dirs.sort(key=lambda x: int(re.split(r"(\d+)", x)[1]))

numdir = 0
for dir in dirs:
    xmls = [
        f
        for f in os.listdir(os.path.join(current_dir, dir))
        if re.match(r"cpt\.\d+\.xml", f)
    ]
    xmls.sort(key=lambda x: re.split(r"(\d+)", x)[1])
    xml = os.path.join(current_dir, dir, xmls[-1])
    dcd = os.path.join(current_dir, dir, DCDNAME)

    path_to_output_file = os.path.join(current_dir, dir, OUTFILE)

    tree = et.parse(xml)
    root = tree.getroot()
    box = root.find(".//box")
    lx = float(box.get("lx"))
    ly = float(box.get("ly"))
    lz = float(box.get("lz"))
    box_size = np.array([lx, ly, lz], dtype=float)

    U = mda.Universe(xml, dcd)
    P = U.select_atoms("type P")
    C = U.select_atoms("type C")
    H = U.select_atoms("type H")
    CH = C + H

    with open(path_to_output_file, "w") as f:
        f.write(f"t\ttrain\ttail\tloops\trate\n")

    t = 1000 * numdir

    for ts in U.trajectory[1:]:
        t += 1
        Pxyz = P.positions
        CHxyz = CH.positions
        fold_back(Pxyz, box_size)
        fold_back(CHxyz, box_size)
        train, tail, loops = find(Pxyz, CHxyz, box_size)
        tot_train.append(train)
        tot_tail.append(tail)
        tot_loops.append(loops)

        with open(path_to_output_file, "a") as f:
            f.write(f"{t*TIMESTEP}\t{train}\t{tail}\t{loops}\t{(tail+loops)/(DP*NP)}\n")
