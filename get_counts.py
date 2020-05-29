#!/usr/bin/python
import subprocess
import os
import glob

counts = {'spatial':0, 'pm':0, 'both':0, 'overlap':0}
total = 0
for fil in glob.glob('*.log'):
    print(fil)
    lines = subprocess.check_output(['tail', '-n', '8', fil], universal_newlines=True, text=True)
    # print(lines)
    line_list = lines.split('\n')[:4]
    print(line_list)
    for key, line in zip(counts, line_list):
        print(line.split(' ')[-3].split('/')[0])

        counts[key] += float(line.split(' ')[-3].split('/')[0])
        if key == 'spatial':
            total += float(line.split(' ')[-3].split('/')[-1])

for keys in counts:
    print(f"{keys} pass rate :  {counts[keys]}/{total} = {counts[keys]/total}")
