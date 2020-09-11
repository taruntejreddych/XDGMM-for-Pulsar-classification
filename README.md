# XDGMM-for-Pulsar-classification
Collected the P, P_dot data from https://www.atnf.csiro.au/research/pulsar/psrcat/ and applied XDGMM after filtering the fliers from the data.
The ellipses are the boundaries which classify the pulsars into different regions based on their locations and emitting spectra. 

pulsar_data contains the pulsar P and Pdot data which are taken from atnf webpage. 
xdgmm_updt contains the code which classifies this data into different regions.
The BIC scores are calculated for different number of model components, and it is found to be least for 6 components. Hence, 6 ellipses have been taken for the model fitting and the result is plotted.

Many data points for Pdot were negative, they were ignored while plotting (Pdot is time derivative and plotting negative time doesn't give any intuition about the boundary regions).
