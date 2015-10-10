% LBL_Evaluate  Main function to evaluate a language model on a dataset
%
% Syntax:
%   [meter, L] = LBL_Evaluate(model, dataset, params, [meter])
% Inputs:
%   model:   struct containing the model being evaluated
%   dataset: struct containing the dataset used for evaluation
%   params:  struct containing the parameters
%   meter:   struct containing the evaluation meter
% Outputs:
%   meter:   struct containing the updated evaluation meter
%   L:       likelihood vector of length <n_nGrams>

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

function [meter, L] = LBL_Evaluate(model, dataset, params, meter)

% Prepare the order of the evaluation dataset
dataset = LBL_Dataset_Init(dataset, params, 0);
if ((nargin < 4) || isempty(meter))
  meter = LBL_Meter_Update('meter_evaluate');
end

% Switch to evaluation mode
params.eval_mode = 1;


% Evaluate the statistics per block
n_words = dataset.n_samples;
L = zeros(1, n_words);
eR = zeros(1, n_words);
fprintf(1, 'Evaluation on %4d batches:\n', dataset.n_batches);
for k = 1:dataset.n_batches

  % Get batch of data
  [wTarget_k, wHistory_k, xHistory_k, thetas_k, indK] = ...
    LBL_Dataset_fprop(dataset, params, k);

  % E-step to produce the log-likelihood and word probabilities
  [L_k, eR_k, ppx_k, dummy, P_k] = ...
    LBL_Estep(wTarget_k, wHistory_k, xHistory_k, model.R, thetas_k, ...
    model, params);
  L(indK) = L_k;
  eR(indK) = eR_k;

  % Trace
  if (mod(k, 10) == 0)
    fprintf(1, 'o');
  else
    fprintf(1, '.');
  end
  if (params.trace_predictions)
    LBL_Model_Trace(wHistory_k, wTarget_k, P_k, model);
  end
end

% Store results in the meter
ppx = LBL_LogLikelihood_Perplexity(L);
meter = LBL_Meter_Update(meter, sum(L), sum(eR), ppx);
fprintf(1, '\nEvaluation: perplexity %g on %d words (%d OOV)\n', ...
  ppx, n_words, sum(isnan(L)));
