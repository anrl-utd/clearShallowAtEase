import sys
import csv

with open(sys.argv[2], mode="w") as out:
    out = csv.writer(out, delimiter=',')
    with open(sys.argv[1]) as log:
        model_stats = ['unbalanced accuracy average','unbalanced recall average','unbalanced precision average',
        'balanced accuracy average','balanced recall average','balanced precision average']
        for i, line in enumerate(log):
            if i % 266 == 258:
                # 26th line
                model_stats[0] = line[:-1]
            elif i % 266 == 259:
                # 26th line
                model_stats[1] = line[:-1]  
            elif i % 266 == 260:
                # 26th line
                model_stats[2] = line[:-1]
            elif i % 266 == 261:
                # 26th line
                model_stats[3] = line[:-1]
            elif i % 266 == 262:
                # 26th line
                model_stats[4] = line[:-1]
            elif i % 266 == 263:
                # 26th line
                model_stats[5] = line[:-1]
            elif (i/266.0).is_integer():
                out.writerow(model_stats)
                                 