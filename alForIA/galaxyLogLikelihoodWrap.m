function [logLiklVals] = galaxyLogLikelihoodWrap(evalAtPts, lowestLogLiklVal)
% evalAtPts is an numPtsxnumDims array. Each row is a point at which we want to
% evaluate the log Likelihood. If you are querying at one point, just pass a row
% vector (1 x numDims) containing the point.

  % Prelims
  numPts = size(evalAtPts, 1);
  numDims = size(evalAtPts, 2);

  logLiklVals = zeros(numPts, 1);
  % Now call the simulator iteratively.
  for iter = 1:numPts
    
    % First write to file
    currEvalPt = evalAtPts(iter, :);
    inFile = sprintf('sim/queryIn_%s', datestr(now, 'mmdd-HHMMSS'));
    outFile = sprintf('sim/liklOut_%s', datestr(now, 'mmdd-HHMMSS'));
    save(inFile, 'currEvalPt', '-ascii');

    % Now Call the simulator
    % @ Ying TODO: Add your simulator here. For now, I've created a directory
    % named galaxySim but if that's not feasible you'll have to add the full
    % path here.
    system('module load python27');
%     callSimCmd = sprintf('python sim/dummy.py %s %s', inFile, outFile);
    callSimCmd = sprintf( ['export LD_LIBRARY_PATH=/opt/python27/lib; ', ...
      'python sim/loglike.py %s %s > garbageCan'], inFile, outFile);
    system(callSimCmd);

    % Now read the output and store the result
    outVal = load(outFile);
    logLiklVals(iter) = max(outVal, lowestLogLiklVal); 
    fprintf('True Loglikl: %0.2f, ret: %0.2f, query pt: %s\n', ...
      outVal, logLiklVals(iter), mat2str(currEvalPt) );

    % Delete the files
    delete(inFile)
    delete(outFile);
  end

end
