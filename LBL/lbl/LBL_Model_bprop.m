% LBL_Model_bprop  Back-propagate the model on a minibatch to get gradients
%
% Syntax:
%   [dL_dzVocabulary, wAll, model] = ...
%     LBL_Model_bprop(P, zBar, zHistory, zVocabulary, zHidden, ...
%                     wTarget, wHistory, xHistory, thetas, model, params)
% Inputs:
%   P:           matrix of size <dim_w> x <n_samples> of word probabilities
%   zBar:        matrix of size <dim_z> x <n_samples> of latent predictions
%   zHistory:    matrix of size <dim_z> x <n> x <n_samples> of latent hist.
%   zVocabulary: matrix of size <dim_z> x <dim_w> of vocabulary embeddding
%   zHidden:     matrix of size <dim_h> x <n_samples> of nonlinear
%                activations, or a cell array of <dim_k> such matrices
%   wTarget:     vector of size 1 x <n_samples> of target word indexes
%   wHistory:    matrix of size <n> x <n_samples> of n-gram word indexes
%   xHistory:    matrix of size <dim_x> x <n> x <n_samples> of features
%   thetas:      matrix of topic mixtures of size <dim_k> x <n_samples>
%   model:       struct containing the model
%   params:      struct containing the parameters
% Outputs:
%   dL_dzVocabulary: matrix of size <dim_z> x <dim_w> of word gradients
%   wAll:            vector of indexes of updated words
%   model:           struct containing the model with updated gradients

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

function [dL_dzVocabulary, wAll, model] = ...
  LBL_Model_bprop(P, zBar, zHistory, zVocabulary, zHidden, ...
  wTarget, wHistory, xHistory, thetas, model, params)


% Number of predictions made
n_samples = size(zBar, 2);


% ---------------------------------------------------------------
% Compute the gradients of the log-likelihood for all predictions
% ---------------------------------------------------------------

if (params.approx_ll)

  dL_dzBar = zeros(model.dim_zw, n_samples);
  dL_dzTarget = zeros(model.dim_zw, n_samples);
  model.dL_dbw = zeros(model.dim_w, 1);

  % Coefficient weighting on the probability estimates
  delta = params.dim_w / params.n_approx_ll;
  for k = 1:n_samples

    % Get the prediction and predictive distribution for <k>-th prediction
    zBar_k = zBar(:, k);
    w_k = wTarget(k);
  
    % Given the approximate log-likelihood <p> for one prediction <zBar>,
    % compute the gradients to the embeddings <z> and <zBar>
    indNeighb_k = model.graphNeighb(w_k, :);
    p_w_k = P(1, k);
    pNeighb_k = P(:, k);
    [dL_dzTarget(:, k), dL_dzBar(:, k), dL_dbw_k] = ...
      LBL_LogLikelihood_BiLinearApprox_bprop(zVocabulary, zBar_k, ...
      p_w_k, pNeighb_k, w_k, indNeighb_k, delta);

    % Accumulate the gradients on the word biases
    model.dL_dbw(w_k) = model.dL_dbw(w_k) + dL_dbw_k;
  end
else
  
  % Block operation
  [dL_dzOutput, dL_dzBar, model.dL_dbw] = ...
    LBL_LogLikelihood_BiLinear_bprop(zVocabulary, zBar, P, wTarget);
end


% ------------------------------------------------------------------------
% Back-propagate the gradients on the predictions <zBar> to the predictors
% ------------------------------------------------------------------------

if (nargout > 1)

  % ... 1) to get the gradients w.r.t. linear predictor's parameters
  if (params.use_lin)
    [dL_dzHistory, model] = ...
      LBL_Module_LinearDynamics_bprop(zHistory, dL_dzBar, thetas, model);
  end
  if (model.dim_h > 0)
    % ... 2) to get the gradients w.r.t. nonlinear predictor's parameters
    [dL_dzHistory_nonlinear, model] = ...
      LBL_Module_NonLinearDynamics_bprop(zHistory, zHidden, ...
      dL_dzBar, thetas, model);
    % Gradients onto linear and nonlinear histories add up
    if (params.use_lin)
      dL_dzHistory = dL_dzHistory + dL_dzHistory_nonlinear;
    else
      dL_dzHistory = dL_dzHistory_nonlinear;
    end
  end
else

  % ... or 1) only the gradients w.r.t. the embedding history (linear)
  if (params.use_lin)
    dL_dzHistory = ...
      LBL_Module_LinearDynamics_bprop(zHistory, dL_dzBar, thetas, model);
  end
  if (model.dim_h > 0)
    % ... 2) to get the gradients w.r.t. nonlinear predictor's parameters
    dL_dzHistory_nonlinear = ...
      LBL_Module_NonLinearDynamics_bprop(zHistory, zHidden, dL_dzBar, ...
      thetas, model);
    % Gradients onto linear and nonlinear histories add up
    if (params.use_lin)
      dL_dzHistory = dL_dzHistory + dL_dzHistory_nonlinear;
    else
      dL_dzHistory = dL_dzHistory_nonlinear;
    end
  end
end

% Gradients onto the features
if (model.dim_zx > 0)

  % Separate the gradients onto word embeddings from gradients on features
  [dL_dzHistory, dL_dzxHistory] = ...
    LBL_Module_History_bprop(dL_dzHistory, params);

  % Back-propagate the gradients onto the feature embedding matrix
  if (model.dim_zx ~= model.dim_x)
    model = LBL_Module_Features_bprop(xHistory, dL_dzxHistory, model);
  end
end


% -------------------------------------------------
% Merge all the gradients onto the latent variables
% -------------------------------------------------

% Select the indexes of all the words involved in targets and histories
wAll = union(wTarget, wHistory(:));

% Merge the gradients on the latent embeddings, sorted by word index
% dL_dzVocabulary = zeros(model.dim_zw, model.dim_w);
dL_dzVocabulary = dL_dzOutput;
for k = 1:n_samples
  for n = 1:model.n
    dL_dzVocabulary(:, wHistory(n, k)) = ...
      dL_dzVocabulary(:, wHistory(n, k)) + dL_dzHistory(:, n, k);
  end
end


% --------------------------------------------
% Add graph constrains on the latent variables
% --------------------------------------------

if (params.gamma > 0)
  dL_dzVocabulary = ...
    LBL_Module_Graph_bprop(wAll, zVocabulary, dL_dzVocabulary, params);
end


% ----------------------------------------------------------
% Back-propagate the gradients on the history to the encoder
% ----------------------------------------------------------

if (nargout > 1)
  model = LBL_Module_Encoder_bprop(wAll, dL_dzVocabulary(:, wAll), model);
end
