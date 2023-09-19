import numpy as np 
from packet_environment import packets
import matplotlib.pyplot as plt
import imageio

"""
Script for converting a packet plume simulation into a movie. Note that typically writing the movie can take about 2s per frame with provided resolutions. 
Choose spatial and temporal resolution depending on your needs. 
"""

fps = 30 #frame rate for movie
run_time = 30 #video of 30s
px_per_mm = 4
max_x = 300
max_y = 180


plume = packets(delta_t = 1/fps, init_intensity = 10740.4)

x_list = np.linspace(0, max_x, int(px_per_mm*max_x))
y_list = np.linspace(0, max_y, int(px_per_mm*max_y))
xx, yy = np.meshgrid(x_list, y_list)
all_x = xx.flatten()
all_y = yy.flatten()

pos_arr = np.zeros((len(all_x), 2))
pos_arr[:,0] = all_x
pos_arr[:,1] = all_y

num_steps = int(fps*run_time)
#num_steps = 5 #if you want to just run a few steps to see a frame and make sure things are working

output_file = 'default_simulated_plume.mp4'
writer = imageio.get_writer(output_file, fps=fps)


for i in range(0, num_steps):

	plume.evolve_packets()
	odors = plume.compute_sig(pos_arr)
	frame = odors.reshape(np.shape(xx))
	frame[frame>255] = 255
	

	#The following block exists if you want to visualize a frame

	"""

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(frame, extent = [0,max_x,0,max_y])
	plt.show()

	"""
	
	frame = frame.astype(np.uint8)
	writer.append_data(frame)


writer.close()

