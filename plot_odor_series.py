import numpy as np 
from packet_environment import packets
import matplotlib.pyplot as plt

""" 
Script that shows how to simulate example odor series and plots the result 
"""

points = np.array([[150,90], [150,60]])
run_time = 30

plume = packets()
series = plume.get_series_at_locations(points = points, run_time = run_time)

t = plume.delta_t * np.arange(np.shape(series)[1])

y_lim = [0,15]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t, series[0], label = 'series at center of plume')
plt.legend()
ax.set_ylabel('odor concentration (au)')
ax.set_ylim(y_lim)
ax = fig.add_subplot(212)
ax.set_ylim(y_lim)
ax.set_ylabel('odor concentration (au)')
plt.plot(t, series[1], label = 'series away from plume in crosswind direction', c = 'orange')
plt.legend()
plt.xlabel('time (s)')
plt.show()


plume2 = packets(rate = 25, dw_speed=60, cw_speed=20, r0=10) #running in slower plume
new_series = plume2.get_series_at_locations(points, run_time=run_time)

#y_lim = [0,10]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t, series[0], label = 'series at center of first plume')
plt.legend()
ax.set_ylabel('odor concentration (au)')
ax.set_ylim(y_lim)
ax = fig.add_subplot(212)
ax.set_ylim(y_lim)
ax.set_ylabel('odor concentration (au)')
plt.plot(t, new_series[0], label = 'series at center of second plume', c = 'orange')
plt.legend()
plt.xlabel('time (s)')
plt.show()
