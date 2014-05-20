% v15: Galaxy Colouring

clear all;
close all;
rng('shuffle'); % Shuffle the seed for Matlab's RNG
addpath GPLibkky/
addpath UCLibkky/
addpath LipLibkky/
addpath MCMCLibkky/
addpath ABCLibkky/
addpath helper/

% Load all constants
loadConstants;

for experiment_iter = 1:NUM_EXPERIMENTS

  fprintf('Uncertainty Reduction\n');
  UncertaintyReductionForIA;

  fprintf('Max Band Point\n');
  MaxBandPointForIA;

end

