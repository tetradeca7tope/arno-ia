% A file to set all parameters for the problem

% Problem Dependent Constants
numDims = 2;
problemSpaceBounds = [0 90; 0 90];
lowestLogliklVal = -1e8; % TODO @Ying Need an estimate for this
logLiklRange = 5e7; % TODO @Ying: need an estimate for this.

% Set up problem class 
paramSpaceBounds = repmat([0 1], numDims, 1);
gcExp = GCExperiment(problemSpaceBounds, lowestLogliklVal);
evalLogJoint = @(arg) gcExp.normCoordLogJointProbs(arg);

DEBUG_MODE = false;
% DEBUG_MODE = true;
if ~DEBUG_MODE
  NUM_AL_ITERS = 500;
  NUM_EXPERIMENTS = 1;
else
  NUM_AL_ITERS = 2;
  NUM_EXPERIMENTS = 1;
end

% Parameters for fitting the GP
noiseLevelGP = logLiklRange / 100;
cvCostFunc = @(y1, y2) (exp(y1) - exp(y2)).^2;

% Parameters for ACtive Learning
numALCandidates = 1000;

% Constants for MaxBandPoint
initLipschitzConstant = 1e8;

% Parameters for Uncertainty Reduction
alBandwidth = 0.4 * NUM_AL_ITERS ^ (-1 /(1.3 + numDims));
alScale = logLiklRange;

% Parameters for MCMC
NUM_MCMC_SAMPLES = 10 * NUM_AL_ITERS;
mcmcProposalStd = 0.3;
mcmcInitPt = 0.5*ones(numDims, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Other Ancillary Variables

timestamp = datestr(now, 'mm:dd-HH:MM:SS');
UC_FUNCHANDLE_FILE = sprintf('results/ucfh_%s.mat', timestamp);
UC_EXP_FILE = sprintf('results/ucexp_%s.mat', timestamp);
MBP_FUNCHANDLE_FILE = sprintf('results/mbpfh_%s.mat', timestamp);
MBP_EXP_FILE = sprintf('results/mbpexp_%s.mat', timestamp);

