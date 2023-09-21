import numpy as np
import scipy.spatial

"""
Implementing simple simulated 2D odor plumes, based on the idea in Farrell et al. 2002, where an odor plume consists of Gaussian concentration packets 
released in a Poisson process and executing a biased random walk. In Farrell et al. the walk is a 2D Wiener Process with drift. 
Here, we allow for Gaussian packets to execute a variety of random walks, in accordance with different models of turbulent flows.
All units for lengths are typically taken to be in mm
All units for time are typically taken to be in seconds
"""


class packets:

	def __init__(self, rate:float = 10, dw_speed:float = 150, init_intensity:float = 1074.7, cw_type:str = 'O-U', eddy_D:float = 0.1*(10**4), 
			   r0:float = 4, packet_D:float = 10, source_x:float = 0, source_y:float = 90, max_x:float = 600, delay_steps:int = 240, delta_t:float = 1/60, 
			  rand_gen:np.random.mtrand.RandomState = np.random.RandomState(0), signal_noise:bool = False, noise_std:float = None, corr_lambda:float = 0.5, cw_speed:float = 45):

		self.rate = rate #packet Poisson release rate
		self.r0 = r0 #initial packet radius
		self.packet_D = packet_D #packet growth diffusivity
		self.source_x = source_x #x-coordinate of odor source
		self.source_y = source_y #y-coordinate of odor source
		self.init_intensity = init_intensity #initial intensity of Gaussian packet peak, arbitrary units. Note that this is I_0 in the formula I_0/(pi*sigma^2), so that the 2D Gaussian density integrates to I_0. Set by default so that at an r0 of 4mm, at 1.75 sigma the concentration is 1 unit.
		self.max_x = max_x #maximum x-coordinate after which packets are ignored, to speed up the simulation. 
		self.rand_gen = rand_gen #np.random.RandomState() object. Could be updated to use default_rng.
		self.delta_t = delta_t #time-step of simulation

		self.packet_xs = np.array([source_x]) #array of packet centroid x-coordinates. Initialized so there is a packet at the source at t=0.
		self.packet_ys = np.array([source_y]) #same as above but for y-coordinates.
		self.packet_durations = np.array([0]) #tracks how long a packet has existed, in order to set its size. 
		self.packet_sizes = np.array([r0]) #starts with initial size r0.

		#cw_type is how packet positions are perturbed. Typically thought of as perturbations only in the crosswind axis, but Gaussian, for example, affects downwind position also.

		if cw_type == 'Gaussian':
			self.eddy_D = eddy_D #turbulent diffusivity-sets size of perturbations. 
		
		if cw_type == 'telegraph': #whether packets undergo a telegraph process in the crosswind direction (i.e. cw velocity switches with a Poisson rate from +v to -v)
			self.corr_lambda = corr_lambda #switching rate
			self.cw_speed = cw_speed #crosswind speed
			self.packet_signs = np.array([1]) #tracks which packets start with positive velocity, which with negative. 

		elif cw_type == 'O-U': #whether crosswind velocity undergoes an Ornstein-Uhlenbeck process. See Pope's Simple Models of Turbulent Flows (2011).
			self.corr_lambda = corr_lambda #reciprocal of Lagrangian correlation timescale (so 1/T_L in Pope's notation)
			self.cw_speed = cw_speed #u' in Pope's notation
			self.packet_vel_y = np.array([0]) #starts with 0 velocity in crosswind direction. 


		self.cw_type = cw_type
		self.dw_speed = dw_speed #packet downwind propagation speed.
		self.noise = signal_noise #whether we want to add Gaussian white noise to the signal.
		self.noise_std = noise_std if noise_std else None 

		for i in range(0, delay_steps): #this generates and evolves packets for delay_steps, so that a plume has formed before the simulation may be called.
			_, _ = self.evolve_packets()


	def evolve_packets(self): 
		
		"""
		Generates and evolves packets by one time step. 
		Returns packet_pos_mat, a matrix of packet positions where the first column is x-coordinates and the second is y-coords. 
		Also returns packet sizes. Together with packet positions, these can be used to compute the signal at a specific point in space-see compute_sig. 
		Can be called in loop in a run script to simulate a dynamic plume.	
		"""

		#determines whether a new packet is created, and appends appropriate attributes if so. 
		
		rand_unif_1 = self.rand_gen.random_sample(1)
		prob = 1 - np.exp(-self.rate*self.delta_t)
		if rand_unif_1 < prob:
			self.packet_xs = np.append(self.packet_xs, self.source_x)
			self.packet_ys = np.append(self.packet_ys, self.source_y)
			self.packet_durations = np.append(self.packet_durations, 0)
			if self.cw_type == 'telegraph':
				rand_unif_2 = self.rand_gen.random_sample(1)
				sign = 2*(rand_unif_2<0.5)-1
				self.packet_signs = np.append(self.packet_signs, sign)
			elif self.cw_type == 'O-U':	
				self.packet_vel_y = np.append(self.packet_vel_y, 0)
			
			self.packet_sizes = np.append(self.packet_sizes, self.r0)


		self.packet_xs = self.delta_t*self.dw_speed + self.packet_xs #advection downwind due to mean flow

		#removes packets that have exceeded max_x. 
		
		if self.max_x != None:
			bools = self.packet_xs < self.max_x
			self.packet_xs = self.packet_xs[bools]
			self.packet_ys = self.packet_ys[bools]
			self.packet_durations = self.packet_durations[bools]
			self.packet_sizes = self.packet_sizes[bools]
			if self.cw_type == 'telegraph':
				self.packet_signs = self.packet_signs[bools]
			elif self.cw_type == 'O-U':
				self.packet_vel_y = self.packet_vel_y[bools]

		#generates appropriate crosswind perturbations.
		
		if self.cw_type == 'Gaussian':
			perts = self.rand_gen.normal(loc = 0, scale = np.sqrt(2*self.eddy_D*self.delta_t), size = (len(self.packet_ys), 2))
		elif self.cw_type == 'Cauchy':
			perts = np.sqrt(self.eddy_D*self.delta_t) * self.rand_gen.standard_cauchy((len(self.packet_ys),2))
		elif self.cw_type == 'telegraph':
			flip_rands = self.rand_gen.uniform(size = len(self.packet_signs))
			flip_bool = flip_rands < self.delta_t*self.corr_lambda
			self.packet_signs[flip_bool] = -1*self.packet_signs[flip_bool]
			perts = np.zeros((len(self.packet_signs),2))
			perts[:,1] = self.cw_speed*self.packet_signs*self.delta_t
		elif self.cw_type == 'O-U':
			perts = np.zeros((len(self.packet_xs),2))
			du_y = -self.packet_vel_y*self.delta_t*self.corr_lambda + np.sqrt(2*(self.cw_speed**2)*self.delta_t*self.corr_lambda)*self.rand_gen.normal(loc = 0, scale = 1, size = len(self.packet_ys))
			self.packet_vel_y = self.packet_vel_y + du_y
			perts[:,1] = self.packet_vel_y*self.delta_t


		#updates and returns packet positions and sizes
		
		self.packet_ys = self.packet_ys + perts[:,1]
		self.packet_xs = self.packet_xs + perts[:,0]
		self.packet_durations = self.packet_durations + self.delta_t
		self.packet_sizes = (self.r0 ** (2) + 4*self.packet_D*self.packet_durations)**0.5


		self.packet_pos_mat = np.zeros((len(self.packet_xs), 2))
		self.packet_pos_mat[:,0] = self.packet_xs
		self.packet_pos_mat[:,1] = self.packet_ys

		return self.packet_pos_mat, self.packet_sizes



	def compute_sig(self, all_points):

		"""
		Computes odor signal at a given set of locations (all_points). all_points is expected to be an array of size (n,2),
		where first column indicates x-coordinate and second indicates y-coordinate.
		"""

		all_distances = scipy.spatial.distance_matrix(all_points, self.packet_pos_mat) #creates a matrix of size (num_points, num_packets) and stores distance from point i to packet j
		scaled_all_distances = all_distances/(self.packet_sizes[None,:])
		gaussian_part = np.exp(-(scaled_all_distances)**2)
		packet_prefactor = self.init_intensity/(np.pi*self.packet_sizes**2)
		all_signals_per_packet = gaussian_part * packet_prefactor[None, :]
		all_total_signals = np.sum(all_signals_per_packet, axis = 1)

		return all_total_signals


	def compute_sig_lr(self, left_points, right_points):

		"""
		Computes odor signal at a series of points. Split into left and right points because often 
		we simulate agents with a set of left odor sensors and a set of right odor at spatially separated locations and want to know 
		the average left and right odor across each sensor, individually. 
		"""

		all_points = np.vstack((left_points, right_points))
		all_signals = compute_sig(all_points)
		total_left_sig = all_signals[0:len(left_points[:,0])]
		total_right_sig = all_signals[len(left_points[:,0]):]

		left_sig = np.mean(total_left_sig) #averages across left sensors
		right_sig = np.mean(total_right_sig) #averages across right sensors

		if self.noise:
			sig_noise = self.rand_gen.normal(loc = 0, scale = self.noise_std, size = (2))
			left_sig = left_sig + sig_noise[0]
			right_sig = right_sig + sig_noise[1]
			left_sig = left_sig * (left_sig > 0) 
			right_sig = right_sig * (right_sig > 0)

		return left_sig, right_sig


	def get_series_at_locations(self, points, run_time):

		"""
		returns a matrix of size (num_points, num_timesteps) of odor concentrations at each point
		"""
		
		num_steps = np.rint(run_time/self.delta_t).astype(int)
		all_series = np.zeros((np.shape(points)[0], num_steps))


		for i in range(0,num_steps):
			self.evolve_packets()
			all_sigs = self.compute_sig(points)
			all_series[:,i] = all_sigs


		return all_series














