% LBL_Model_fprop  Forward propagation of the model on a minibatch
%
% Syntax:
%   [P, L, eR, zBar, zHistory, zHidden] = ...
%     LBL_Model_fprop(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
%                     model, params)
% Inputs:
%   wTarget:     vector of size 1 x <n_samples> of target word indexes
%   wHistory:    matrix of size <n> x <n_samples> of n-gram word indexes
%   xHistory:    matrix of size <dim_x> x <n> x <n_samples> of features
%   zVocabulary: matrix of size <dim_z> x <dim_w> of vocabulary embeddding
%   thetas:      matrix of topic mixtures of size <dim_k> x <n_samples>
%   model:       struct containing the model after 1 learning step
%   params:      struct containing the parameters
% Outputs:
%   P:        matrix of size <dim_w> x <n_samples> of word probabilities
%   L:        vector of length <n_samples> of log-likelihoods
%   eR:       vector of length <n_samples> of observation errors
%   zBar:     matrix of size <dim_z> x <n_samples> of latent predictions
%   zHistory: matrix of size <dim_z> x <n> x <n_samples> of latent history
%   zHidden:  matrix of size <dim_h> x <n_samples> of nonlinear activations
%             or cell array of <dim_k> such matrices

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

function [P, L, eR, zBar, zHistory, zHidden] = ...
  LBL_Model_fprop(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
  model, params)


% Embedding error (TODO)
eR = 0;


% Extract the history from the latent variables
if isempty(xHistory)

  % Use only the history of word embeddings
  zHistory = LBL_Module_History_fprop(wHistory, zVocabulary, model);
else
  
  % Embed the features
  zxHistory = LBL_Module_Features_fprop(xHistory, model);
  % Combine the feature embeddings with the word embeddings for the history
  zHistory = ...
    LBL_Module_History_fprop(wHistory, zVocabulary, model, zxHistory);
end


% Predict the embedding of next word given the embedding history:
% 1) Using the linear module
if (params.use_lin)
  zBar = LBL_Module_LinearDynamics_fprop(zHistory, thetas, model);
end
if (model.dim_h > 0)
  % 2) Optionally, using the nonlinear module
  [zBar_nonlinear, zHidden] = ...
    LBL_Module_NonLinearDynamics_fprop(zHistory, thetas, model);
  if (params.use_lin)
    zBar = zBar + zBar_nonlinear;
  else
    zBar = zBar_nonlinear;
  end
else
  zHidden = [];
end

% Evaluate the log- and likelihoods <L> and <P> of all the words 
% in the vocabulary given the prediction <zBar> and the embedding 
% of the vocabulary <zVocabulary>
if (params.approx_ll && ~params.eval_mode)
  [L, P] = ...
    LBL_LogLikelihood_BiLinearApprox_fprop(wTarget, zVocabulary, zBar, ...
    model);
else
  [L, P] = ...
    LBL_LogLikelihood_BiLinear_fprop(wTarget, zVocabulary, zBar, model);
end
