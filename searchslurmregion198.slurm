#!/bin/bash
#SBATCH --job-name=the_search_region198
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=02-00:00:00 ## time format is DD-HH:MM:SS
## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G ## max amount of memory per node you require

##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node

#SBATCH --error=region198.err ## %A - filled with jobid
#SBATCH --output=region198.log ## %A - filled with jobid
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
python3 -u main.py ./candidates/region_list/region198.txt 18