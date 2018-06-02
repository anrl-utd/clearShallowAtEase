import re
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import glob, os

# A script that takes in a training file as input, and outputs a pyplot of val_acc v. epoch
# input is through terminal argument

# Helper functions
def get_val_list(PATH):
	val_list = []
	for line in open(PATH):
		if(line.find(" Validation Accuracy: ")):
			x = line.find(" Validation Accuracy:")
			val_list.append((line[x+23:x+27]))
	return val_list

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check_val_list(val_list):
	for i, s in enumerate(val_list):
		if(is_num(s)):
			if("." not in s):
				val_list[i] = "0"
				continue
			val_list[i] = float(s)
		else:
			val_list[i] = "0"

	val_list = [x for x in val_list if x != "0"]
	return val_list	

def graph_all(files):
	all_val_lists = {}
	for path in files:
		val_list = get_val_list(path)
		val_list = check_val_list(val_list)
		all_val_lists.update({path:val_list})
	return all_val_lists
# ------------------------------------------------------------

# graph
DIR = "/home/sid/rddnn/logs/"
file_list = []
os.chdir(DIR)
for file in glob.glob("*.txt"):
	file_list.append(file)

print(file_list)
all_vals = graph_all(file_list)
legend = []
for key, value in all_vals.iteritems():
	plt.plot(value)
	legend.append(key)

plt.legend(legend, loc='lower right')
plt.xlabel("epoch")
plt.ylabel("Validation accuracy (%)")
plt.savefig(DIR + "graph.png", bbox_inches='tight')
#plt.show()
