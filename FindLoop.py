#!/opt/homebrew/bin/
#dfgdgdfgd
import re
import numba as nb
import numpy as np
import sys
import warnings

NUMCH=10658                                    #total number of lipid head
NP=int(sys.argv[2])                            #number of polymer chains
DP=int(sys.argv[1])                            #number of monomer on each chain
ADSDIS=2.5                                     #max adsorption distance
RANGE=2.0                                      #neighbor lipid on xy plane
LX=80
LY=80
LZ=40


RANGESQ=RANGE**2
NUMP=NP*DP
HALFLX=LX/2
HALFLY=LY/2
HALFLZ=LZ/2

Px,Py,Pz=[],[],[]
CHx,CHy,CHz=[],[],[]


warnings.filterwarnings("ignore")

def readfile(file):
    for line in file:
        row=re.split(r"\s+",line)
        if row[0]=='P':
            Px.append(float(row[1]))
            Py.append(float(row[2]))
            Pz.append(float(row[3]))
        elif row[0]=='C' or row[0]=='H':
            CHx.append(float(row[1]))
            CHy.append(float(row[2]))
            CHz.append(float(row[3]))


@nb.njit
def fold_back(Px,Py,Pz,CHx,CHy,CHz):                        #calculate the coordiante of non-continuous
    for i in range(0,len(Px)):
        Px[i]=(Px[i]+HALFLX)%LX-HALFLX
        Py[i]=(Py[i]+HALFLY)%LY-HALFLY
        Pz[i]=(Pz[i]+HALFLZ)%LZ-HALFLZ
    for i in range(0,len(CHx)):
        CHx[i]=(CHx[i]+HALFLX)%LX-HALFLX
        CHy[i]=(CHy[i]+HALFLY)%LY-HALFLY
        CHz[i]=(CHz[i]+HALFLZ)%LZ-HALFLZ


@nb.njit
def find(Px,Py,Pz,Cx,Cy,Cz):
    train,tail,loops=[],[],[]
    
    for loop in range(1,int(len(Px)/NUMP)+1):                       #one loop means one frame in traj.xyz
        dis=[]
        train.append(0)
        tail.append(0)
        loops.append(0)
        
        for i in range(NUMP*(loop-1),NUMP*loop):                    #for each monomer, find the closest lipid head and save the distance
            distancesq=[]
            for j in range(NUMCH*(loop-1),NUMCH*loop):
                rsq=(Cx[j]-Px[i])**2+(Cy[j]-Py[i])**2
                if rsq <= RANGESQ:
                    distancesq.append(rsq+(Cz[j]-Pz[i])**2)
            dis.append(min(distancesq))
            
        for i in range(0,NP):
            chain=dis[i*DP:(i+1)*DP]
            binchain=[1 if x < ADSDIS else 0 for x in chain]
            train[loop-1]+=sum(binchain)
            for j in range(0,DP):
                if binchain[j]==1:
                    tail[loop-1]+=j
                    break
            for j in range(DP-1,-1,-1):
                if binchain[j]==1:
                    tail[loop-1]+=(DP-1)-j
                    break
            loops[loop-1]=NP*DP-train[loop-1]-tail[loop-1]
            
    return train,tail,loops


@nb.njit
def Calc_aver_data(train,tail,loops):
    aver_train=np.average(train)
    aver_tail=np.average(tail)
    aver_loops=np.average(loops)
    return aver_train,aver_tail,aver_loops


with open("traj.xyz","r") as file:
    readfile(file)
            
if len(Px)/NUMP!=1001:
    raise Exception("INPUT FILE IS INVALID")

fold_back(Px,Py,Pz,CHx,CHy,CHz)

train,tail,loops=find(Px,Py,Pz,CHx,CHy,CHz)

aver_train,aver_tail,aver_loops=Calc_aver_data(train,tail,loops)

with open("FindLoop.dat","w") as f:
    f.write(f"train_average: {aver_train}\ntail_average:  {aver_tail}\nloops_average: {aver_loops}\n\n")
    f.write(f" frame \t train \t tail \t loops \n")
    for i in range(0,1001):
        f.write(f"{i+1:^7}\t{train[i]:^7}\t{tail[i]:^6}\t{loops[i]:^7}\n")