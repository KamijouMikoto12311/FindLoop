## 1. Running directly in the directory

#### Application:

python3 FindLoop.py N_monomer N_polymer

#### Params:

N_monomer: number of monomer
N_polymer: number of polymer

## 2. Running on HPC with slurm

#### Application:

sbatch FindLoop.sbatch
<span style="color: red;">Remember to change the params in FindLoop.sbatch </span>

#### Params:

DP: Number of monomer
NP: Number of polymer
