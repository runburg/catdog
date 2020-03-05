#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune parameters using simulated annealing.

Author: Jack Runburg
Date: 04-09-2019 18:39


"""
import glob
import numpy as np
from the_search.dwarf import Dwarf
from dsph_search.the_search.tuning.annealing import truth_to_power, OptimizeCutParameters

# parameter dict
params = {'test_area': 10, 'test_percentage': 0.179376451145657, 'num_maxima': 8, 'density_tolerance': 1.362830538392538}
params_values = params.values()
params_names = params.keys()
# passing_random_cones = 4

dwarfs = []
randoms = []
for dwa in np.loadtxt('./the_search/tuning/tuning_known_dwarfs.txt', dtype=str, delimiter=','):
    d = Dwarf(dwa[1].astype(np.float), dwa[2].astype(np.float), name=dwa[0])
    for table in glob.glob(f'./the_search/tuning/test_candidates/{d.name}/vots/*.vot'):
        d.load_gaia_table(table)
    dwarfs.append(d)
print('loaded dwarfs')
for ran in np.loadtxt('./the_search/tuning/tuning_random.txt', delimiter=','):
    d = Dwarf(ran[0], ran[1])
    for table in glob.glob(f'./the_search/tuning/test_candidates/{d.name}/vots/*.vot'):
        d.load_gaia_table(table)
    randoms.append(d)
print('loaded randoms')
print(truth_to_power(params))

ocp = OptimizeCutParameters(params, dwarfs, randoms, kind='pm')
ocp.Tmax = 1000.0  # Max (starting) temperature
ocp.Tmin = 5     # Min (ending) temperature
ocp.steps = 2000   # Number of iterations
ocp.updates = 10   # Number of updates (by default an update prints to stdout)
# auto_time = ocp.auto(minutes=0.1, steps=2000)
# ocp.set_schedule(auto_time)
# since our state is just a list, slice is the fastest way to copy
sta, e = ocp.anneal()
print(f"min energy {e} with state {sta}")
