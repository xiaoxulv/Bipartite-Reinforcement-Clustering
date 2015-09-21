__author__ = 'Ariel'

import numpy as np

dict = {}
cset = set()
with open("HW2_dev.gold_standards",'r') as gold:

    for line in gold:
        cset.add(line[5:])