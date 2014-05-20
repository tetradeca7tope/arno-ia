% Script runs UC for Intrinsic Alignments

% Initialization
ucParams.numALCandidates = numALCandidates;
ucParams.lowestLogliklVal = lowestLogliklVal;
ucParams.alBandwidth = alBandwidth;
ucParams.alScale = alScale;
ucParams.gpNoiseLevel = noiseLevelGP;
[ucPts, ucLogProbs] = alGPUncertaintyReduction(evalLogJoint, [], [], ...
  paramSpaceBounds, NUM_AL_ITERS, ucParams);

% Now perform the Regression
ucLogJointEst = regressionWrap(ucPts, ucLogProbs, noiseLevelGP, ...
  lowestLogliklVal, logLiklRange, cvCostFunc);

% Save the results
save(UC_FUNCHANDLE_FILE, 'ucLogJointEst');
save(UC_EXP_FILE, 'ucPts', 'ucLogProbs', 'ucParams', 'noiseLevelGP', ...
  'lowestLogliklVal', 'logLiklRange', 'cvCostFunc');

