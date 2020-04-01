#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make slurm files.

Author: Jack Runburg
Date: 27-09-2019 13:10


"""
import numpy as np
import glob
from the_search.utils import generate_full_sky_cones

# ## For all known dwarfs
# known = np.loadtxt('./dsph_search/the_search/tuning/tuning_known_dwarfs.txt', delimiter=',', dtype=str)
#
# labels = known[:, 0]
# ra_known = known[:, 1].astype(np.float) + np.random.randint(-2, 3, size=len(known))
# dec_known = known[:, 2].astype(np.float) + np.random.randint(-2, 3, size=len(ra_known))
# ##
cone_directory='./candidates/region_list/'

generate_full_sky_cones(region_radius, galactic_plane=18, hemi='both', output_directory=cone_directory)

file_list = glob.glob(cone_directory + '*.txt')

for infile in file_list:
    name = infile.split("/")[-1].strip(".txt")

    with open(f'./searchslurm{name}.slurm', 'w') as outfile:
        outfile.write(f'''#!/bin/bash
#SBATCH --job-name=the_search_{name}
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=02-00:00:00 ## time format is DD-HH:MM:SS
## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G ## max amount of memory per node you require

##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node

#SBATCH --error={name}.err ## %A - filled with jobid
#SBATCH --output={name}.log ## %A - filled with jobid
## Useful for remote notification
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=runburg@hawaii.edu

source ~/.bash_profile

module load lang/Python/3.7.2-intel-2018.5.274

pip3 install --user numpy
pip3 install --user matplotlib
pip3 install --user astropy
pip3 install --user astroquery --upgrade
echo "Finished installing packages"
''')
        outfile.write("python3 -u main.py" + infile)
