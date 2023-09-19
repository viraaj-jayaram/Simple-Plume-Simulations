# Simple-Plume-Simulations
Code to run simple odor plume simulations based on Gaussian odor packets executing random walks with drift. The main idea comes from [Farrell et al. 2002](https://link.springer.com/article/10.1023/A:1016283702837) but the code has been written from scratch by Viraaj Jayaram. Other types of random walks based on different models of turbulent dispersion have also been included-see packet_environment.py for different model descriptions.

The scripts are organized as follows: packet_environment.py is a class that allows for odor plumes to be generated and dynamically evolve. plot_odor_series.py and convert_sim_to_movie show how to use this class to get simulated odor time series at different locations and how to create a movie of a simulated plume. example_simulated_plume.mp4 is an example movie made with default class parameters. 

## Dependencies ##

Code uses python3 and numpy only. For plotting and making movies matplotlib and imageio version 2.28 and imageio-ffmpeg version 0.4.8 are also used. 

