from exp_configs import EXP_GROUPS
import gridFunction

E = EXP_GROUPS['gridSearch']
SaveDir = '/var/tmp/osane/code/bachelorarbeit/results/gridSearch'
DataDir = '/var/tmp/osane/code/bachelorarbeit/data'
INCREASER=['constant','multiplicative','scaling']
GAMMAS={
    'constant': [0.01, 0.001, 0.0001, 0.00001],
    'multiplicative': [2.35e19,5.64e13, 8252622217, 105.29],
    'scaling': [1.1, 1.07, 1.05, 1.01]
}

for increaser in INCREASER:
    for gamma in GAMMAS[increaser]:
        gridFunction.trainval_grid(E[0], SaveDir, DataDir, True, 1, increaser, gamma)
