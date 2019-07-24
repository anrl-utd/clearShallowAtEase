high = [0.999, 0.998,0.995,0.99,0.985,0.985,0.98,0.97]

high_mult = 1
for idx in range(len(high)):
    high_mult *= high[idx]
print('No failure', high_mult)
print('High')
for idx in range(len(high)):
    rmv_mult = high_mult / high[idx]
    print(100 * rmv_mult * (1-high[idx]))

med = [0.99, 0.98, 0.94, 0.93,0.9, 0.9, 0.87, 0.87]

med_mult= 1
for idx in range(len(med)):
    med_mult *= med[idx]
print('No failure', med_mult)
print('Med')
for idx in range(len(med)):
    rmv_mult = med_mult / med[idx]
    print(100 * rmv_mult * (1-med[idx]))

low = [0.9,  0.9, 0.8, 0.8, 0.7, 0.6, 0.7, 0.66]

low_mult= 1
for idx in range(len(low)):
    low_mult *= low[idx]
print('No Failure', low_mult)
print('Low')
for idx in range(len(low)):
    rmv_mult = low_mult/ low[idx]
    print(100 * rmv_mult * (1-low[idx]))

# [f1, f2, f3, f4, e1, e2, e3, e4]
print('e3, e4')
print(100 * (high_mult / (high[6] * high[7])) * (1-high[6]) * (1-high[7]))
print(100 * (med_mult/ (med[6] * med[7])) * (1-med[6]) * (1-med[7]))
print(100 * (low_mult/ (low[6] * low[7])) * (1-low[6]) * (1-low[7]))

print('f3, e1')
print(100 * (high_mult/ (high[2] * high[4])) * (1-high[2]) * (1-high[4]))
print(100 * (med_mult/ (med[2] * med[4])) * (1-med[2]) * (1-med[4]))
print(100 * (low_mult/ (low[2] * low[4])) * (1-low[2]) * (1-low[4]))

print('f4, e2')
print(100 * (high_mult / (high[3] * high[5])) * (1-high[3]) * (1-high[5]))
print(100 * (med_mult/ (med[3] * med[5])) * (1-med[3]) * (1-med[5]))
print(100 * (low_mult/ (low[3] * low[5])) * (1-low[3]) * (1-low[5]))

import numpy as np
print('Complete failure')
c_high = [1-high[idx] for idx in range(len(high))]
print('High', 100*np.prod(c_high))
c_med = [1-med[idx] for idx in range(len(med))]
print('Med', 100*np.prod(c_med))
c_low = [1-low[idx] for idx in range(len(low))]
print('Low', 100*np.prod(c_low))
