import matplotlib.pyplot as plt
import numpy as np
import pylab
def load_data(filename):
	File=open(filename,'r')
	#open file in read only mode
	x_axis=[]
	y1_axis=[]
	y2_axis=[]
	time=0
	#create two axis for the data
	for line in File:
		Set = line.split('\t')
		print(Set)
		x_axis.append(time)
		y1_axis.append(float(Set[1]))
		y2_axis.append(float(Set[2]))
		time += 0.004
	return (x_axis,y2_axis)

def plot(x_axis,y2_axis):
	x=np.array(x_axis)
	y2=np.array(y2_axis)
	plt.plot(x,y2,'b',lw='2')
	plt.xlabel(u'Time/s')
	plt.ylabel(u'ECG signal')
	plt.show()

(x_axis,y2_axis,)=load_data('python.txt')
plot(x_axis,y2_axis)
