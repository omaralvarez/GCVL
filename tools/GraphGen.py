import os
import platform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from matplotlib.mlab import griddata
import numpy

#System info
print ('System: ', platform.system())
print ('Arch: ', platform.machine())
print ('OS: ', platform.platform())
print ('Name: ', platform.node())
print ('Processor: ', platform.processor())

print('\n***Starting Testing***\n')

f = open('sys_info.txt', 'w')
f.write('Name: ' + platform.node() + '\n')
f.write('System: ' + platform.system() + '\n')
f.write('Arch: ' + platform.machine() + '\n')
f.write('OS: ' + platform.platform() + '\n')
f.write('System: ' + platform.system() + '\n')
f.write('Processor: ' + platform.processor() + '\n')
f.close()

runs =  1

def plot3D():

    aggDims = [5,7,9,11,13]
    maxDisps = [16,32,64,128]

    #3D Plotting

    x = list()
    y = list()
    z = list()

    for aggDim in aggDims:
        for maxDisp in maxDisps:
            x.append(aggDim)
            y.append(maxDisp)

    #Check results
    counter = 0
    avg_hits = 0.0
    for file in os.listdir('./data'):
        fullPath = os.path.join('./data',file)
        total_lines = 0
        total_data = 0

        if not os.path.isfile(fullPath):
            continue

        if file == '.DS_Store':
            continue

        for line in open(fullPath):
            z.append(float(line.split(',')[-1]))

    AD = numpy.linspace(min(x), max(x))
    MD = numpy.linspace(min(y), max(y))
    X,Y = numpy.meshgrid(AD, MD)
    Z = griddata(x,y,z,AD,MD)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    fig.colorbar(p, shrink=0.5, aspect=5)
    ax.set_title('OpenCL Performance')
    ax.set_xlabel('Aggregation Dimension (px)')
    ax.set_ylabel('Maximum Disparity (px)')
    ax.set_zlabel('Time (s)')

    plt.savefig(os.path.join('bow_perf_ocl_3d_2.pdf'))

#*****Compute*****
plot3D()
