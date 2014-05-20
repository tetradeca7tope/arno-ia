% Script runs MaxBandPoint for Galaxy Coloring

% Initialization
initialPts = [];
initialLogProbs = [];

% Set parameters for MBP
phi = @exp; gradPhi = @exp; % use exponential transformation
alMbpParams.num_iters = 10;
alMbpParams.init_step_size = 1;
% First obtain the points via MBP
[mbpPts, mbpLogProbs, mbpLipConst] = alMaxBandPoint(evalLogJoint, ...
  initialPts, initialLogProbs, phi, gradPhi, initLipschitzConstant, ...
  paramSpaceBounds, NUM_AL_ITERS, alMbpParams);

% Now perform regression on each of these points to obtain the estimates.
mbpLogJointEst = regressionWrap(mbpPts, mbpLogProbs, noiseLevelGP, ...
  lowestLogliklVal, logLiklRange);

% Save the results
save(MBP_FUNCHANDLE_FILE, 'mbpLogJointEst');
save(MBP_EXP_FILE, 'mbpPts', 'mbpLogProbs', 'initialPts', 'noiseLevelGP', ...
  'lowestLogliklVal', 'logLiklRange');

