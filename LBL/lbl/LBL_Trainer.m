% LBL_Trainer  Main function to train a language model on a text dataset
%
% Syntax:
%   [MODEL, METER_TRAIN, METER_XVAL, METER_LEARN] = ...
%     LBL_Trainer(datasetTrain, datasetXval, params, ...
%                 model, meterTrain, meterXval, meterLearn)
% Inputs:
%   datasetTrain: struct containing the dataset used for training
%   datasetXVal:  struct containing the dataset used for cross-validation
%   params:       struct containing the parameters
%   model:        (optional) struct containing the model:
%                 this is useful for re-training a model
%   meterTrain:   (optional) struct containing the training meter
%   meterXVal:    (optional) struct containing the cross-validation meter
%   meterLearn:   (optional) struct containing the learning meter
% Outputs:
%   model:        (global) struct containing the trained model
%   meterTrain:   (global) struct containing the training meter
%   meterXVal:    (global) struct containing the cross-validation meter
%   meterLearn:   (global) struct containing the learning meter

% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
% Version 1.0, New York, 9 June 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [MODEL, METER_TRAIN, METER_XVAL, METER_LEARN, METER_TEST] = ...
  LBL_Trainer(datasetTrain, datasetXval, datasetTest, params, ...
  model, meterTrain, meterXval, meterLearn, meterTest)


% Initialize the model
if (nargin < 4)
  datasetTest = [];
end
if (nargin < 5)
  model = LBL_Model_Init(params, datasetTrain);
end
global MODEL
MODEL = model;

if (params.gamma > 0)
  % Check that the graph is a cell array
  params = ...
    LBL_Module_Graph_Prepare(params, ...
    params.graphFriends, params.graphFiends);
end

% Reset the meters
if (nargin < 6)
  meterTrain = LBL_Meter_Update('meter_train');
  meterXval = LBL_Meter_Update('meter_xval');
  meterLearn = LBL_Meter_Update('meter_learn');
  if ~isempty(datasetTest)
    meterTest = LBL_Meter_Update('meter_test');
  end
end
global METER_TRAIN
global METER_XVAL
global METER_LEARN
global METER_TEST
METER_TRAIN = meterTrain;
METER_XVAL = meterXval;
METER_LEARN = meterLearn;
if ~isempty(datasetTest)
  METER_TEST = meterTest;
else
  METER_TEST = [];
end

% Keep memory of initial learning rates
params.eta_w0 = params.eta_w;
params.eta_r0 = params.eta_r;
params.eta_c0 = params.eta_c;
params.eta_b0 = params.eta_b;
params.eta_a0 = params.eta_a;
params.eta_f0 = params.eta_f;


% Loop on several epochs until convergence
keep_training = 1;
worsening_count = 0;
measure_previous = inf;
epoch = 1;
while (keep_training)

  % Scramble the order of the training dataset
  datasetTrain = LBL_Dataset_Init(datasetTrain, params, 1);

  % Learning passes over the training data
  t0 = clock;
  for k = 1:datasetTrain.n_batches

    % Get the data sample
    [wTarget, wHistory, xHistory, thetas] = ...
      LBL_Dataset_fprop(datasetTrain, params, k);
    fprintf(1, 'Epoch %3d: %4d/%4d:\n', epoch, k, datasetTrain.n_batches);

    % Re-initialize the samples graph (in case of log-likelihood approx)
    if (params.approx_ll)
      MODEL = LBL_LogLikelihood_BiLinearApprox_Init(MODEL, params);
    end
    
    % Learn the current batch
    [MODEL, METER_LEARN] = ...
      LBL_Learn(wTarget, wHistory, xHistory, thetas, MODEL, ...
      params, METER_LEARN);

    % Handle exploding perplexity
    if (isnan(METER_LEARN.ppx_e(end)) || isinf(METER_LEARN.ppx_e(end)))
      SaveModelCrash(params);
      return;
    end
    
    if (mod(k, 100) == 0)
      fprintf(1, 'Current training epoch took %g seconds so far\n', ...
        etime(clock, t0));
    end
  end
  fprintf(1, '1 training epoch took %g seconds\n', etime(clock, t0));


  % Evaluate the performance on the training data
  if (params.eval_ppx_train)
    METER_TRAIN = LBL_Evaluate(MODEL, datasetTrain, params, METER_TRAIN);
  end

  % Evaluate the performance on the x-validation data
  METER_XVAL = LBL_Evaluate(MODEL, datasetXval, params, METER_XVAL);
  measure_current = METER_XVAL.ppx_e(end);

  % Evaluate the performance on the test data
  if ~isempty(datasetTest)
    METER_TEST = LBL_Evaluate(MODEL, datasetTest, params, METER_TEST);
  end

  % Save best model
  MODEL = LBL_Model_SaveBest(MODEL, measure_current, epoch);
  SaveModel(params);
  
  % Convergence (early stopping)
  epoch = epoch + 1;
  if (measure_current < measure_previous)
    worsening_count = 0;
  else
    worsening_count = worsening_count + 1;
    fprintf(1, 'X-validation measure increasing for %d epochs...\n', ...
      worsening_count);
  end
  measure_previous = measure_current;
  keep_training = ((epoch <= params.n_epochs) & (worsening_count < 5));

  % Stop when perplexity blows-up (in a jagged way)
  if (measure_current > 2 * MODEL.measure)
    keep_training = 0;
  end

  % Learning rate annealing
  params.eta_w = params.eta_w * params.eta_anneal;
  params.eta_r = params.eta_r * params.eta_anneal;
  params.eta_c = params.eta_c * params.eta_anneal;
  params.eta_a = params.eta_a * params.eta_anneal;
  params.eta_b = params.eta_b * params.eta_anneal;
  params.eta_f = params.eta_f * params.eta_anneal;
end


% Retrieve the best model and evaluate it
MODEL = LBL_Model_RetrieveBest(MODEL);

if (params.eval_ppx_train)
  % Evaluate the performance on the training data after one epoch
  METER_TRAIN = LBL_Evaluate(MODEL, datasetTrain, params, METER_TRAIN);
end

% Evaluate the performance on the x-validation data
METER_XVAL = LBL_Evaluate(MODEL, datasetXval, params, METER_XVAL);

if ~isempty(datasetTest)
  % Evaluate the performance on the x-validation data
  METER_TEST = LBL_Evaluate(MODEL, datasetTest, params, METER_TEST);
end

% Save last performance
SaveModel(params);


% -------------------------------------------------------------------------
function SaveModel(params)

global METER_TRAIN
global METER_XVAL
global METER_LEARN
global METER_TEST
global MODEL

% Write to a text file
fid = fopen([params.filename '.txt'], 'w');
LBL_Params_Display(params, fid);
if (params.eval_ppx_train)
  LBL_Meter_Display(METER_TRAIN, fid);
end
LBL_Meter_Display(METER_XVAL, fid);
if ~isempty(METER_TEST)
  LBL_Meter_Display(METER_TEST, fid);
end
fclose(fid);

% Save a Matlab file
save([params.filename '.mat'], 'params', 'MODEL', ...
  'METER_XVAL', 'METER_TRAIN', 'METER_LEARN', 'METER_TEST');


% -------------------------------------------------------------------------
function SaveModelCrash(params)

global METER_TRAIN
global METER_XVAL
global METER_LEARN
global METER_TEST
global MODEL

fprintf(1, 'Perplexity is exploding: interrupting the learning...\n');

% Write to a text file
fid = fopen([params.filename '.txt'], 'w');
LBL_Params_Display(params, fid);
LBL_Meter_Display(METER_TRAIN, fid);
LBL_Meter_Display(METER_XVAL, fid);
if ~isempty(METER_TEST)
  LBL_Meter_Display(METER_TEST, fid);
end
fclose(fid);
