% LBL_Params_Init  Initialize the hyperparameters
%
% Syntax:
%   params = LBL_Params_Init(dim_w, dim_z, n, dim_h, dim_x, dim_zw)
% Inputs:
%   dim_w:  dimension of the vocabulary
%   dim_z:  dimension of the latent variables
%   n:      order of the dynamics (including the prediction)
%   dim_h:  dimension of the hidden, nonlinear layer (optional, default=0)
%   dim_x:  dimension of the features (optional, default=0)
%   dim_zw: dimension of latent variables for word embedding
%           (optional, by default it is equal to <dim_z>)
% Outputs:
%   params: struct containing the parameters

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
% Version 1.0, New York, 19 June 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu
%
% (c) 2010, AT&T Labs
%     180 Park Avenue, Florham Park, NJ 07932, USA.

function params = LBL_Params_Init(dim_w, dim_z, n, dim_h, dim_x, dim_zw)

% No hidden layer (and neural net nonlinearity) by default
if (nargin < 4)
  dim_h = 0;
end
if (nargin < 5)
  dim_x = 0;
end
if (nargin < 6)
  dim_zw = dim_z;
end
dim_zx = dim_z - dim_zw;


% Filename
filename = ...
  sprintf('w%d_n%d_z%d_h%d_x%d_%s', dim_w, n, dim_z, dim_h, dim_x, ...
  datestr(now, 'yyyymmdd_HHMMss'));

params = struct('n_epochs', 100, ...
  'open_vocab', 0, ...  % Is the model open-vocabulary? Closed by default
  ... % Dimensions
  'n', n, ...           % Order of n-gram history
  'dim_w', dim_w, ...   % Number of words
  'dim_z', dim_z, ...   % Number of latent variables
  'dim_h', dim_h, ...   % Dimension of nonlinear hidden layer (0=none)
  'dim_x', dim_x, ...   % Dimension of word feature input
  'dim_zw', dim_zw, ... % Number of word-specific latent variables
  'dim_zx', dim_zx, ... % Number of word feature-specific latent variables
  'use_lin', true, ...  % Shall we even use the linear part?
  'use_neu2', true, ... % Shall we use the 2nd layer of the neural net?
  ... % Weights initialization
  'init_r', 'inv_sqrt', ...
  ... % Log-likelihood approximation
  'approx_ll', 0, ...          % Shall we approximate the likelihood?
  'n_neighb_approx_ll', 0, ... % Number of neighbors in the approximation
  'n_approx_ll', 0, ...        % Total number of words used for the approx
  'graphNeighb', [], ...       % Graph of neighbors for each word
  ... % Learning
  ... % (parameters derived from [Mnih & Hinton, 2007, Mnih et al, 2009])
  'len_batch', 1000, ...   % Length of minibatch (for each gradient update)
  'eta_w', 1e-5, ...       % Learning rate for the word bias in the LBL
  'eta_r', 1e-4, ...       % Learning rate for word representation
  'eta_c', 1e-3, ...       % Learning rate for the linear dynamics
  'eta_a', 1e-1, ...       % Learning rate for first layer of NN
  'eta_b', 1e-5, ...       % Learning rate for second layer of NN
  'eta_f', 1e-4, ...       % Learning rate for the feature embedding
  'eta_anneal', 0.99, ...  % Annealing rate of the learning rate
  'lambda_r', 1e-5, ...    % Regularization of word representation
  'lambda_c', 1e-5, ...    % Regularization of linear dynamics
  'lambda_a', 1e-5, ...    % Regularization of linear dynamics
  'lambda_b', 1e-5, ...    % Regularization of linear dynamics
  'lambda_f', 1e-5, ...    % Regularization of feature embedding
  'momentum', 0.5, ...     % Weight of the momentum on parameters (0=none)
  ... % Graph of friends and fiends
  'gamma', 0, ...
  'graphFriends', [], ...
  'graphFiends', [], ...
  ... % Topic model
  'n_topics', 0, ...         % if not 0, then a small number
  'topic_model', 'none', ... % Could be "lda" or "lsi" or "ica"
  ... % Inference of latent variables
  'relax_z', 0, ...  % No real inference
  'eta_z', 1e-1, ... % Inference rate
  ... % Model evaluation
  'eval_model', 0, ...        % By default, not in evaluation mode (learn)
  'eval_ppx_after', 0, ...    % Evaluate (L, ppx) *after* each M-step?
  'eval_ppx_train', 0, ...    % Skip the evaluation of the train data
  'trace_predictions', 0, ... % Print the best n-gram prediction?
  ... % Save results
  'filename', filename);   % Filename where results are saved

% On the ATIS dataset:
% * eta_anneal=0.97 stops later (at lower perplexity) than 0.95
%   eta_anneal=0.99 was good
% * eta_w=eta_r=1e-3 is too much
%   take eta_r=1e-4
% * eta_c of 1e-5 is too low
%   eta_c=1e-3
% * lambda_r of 1e-4 was good
%   take lambda_r=1e-5
% * eta_a=1e-1
% * eta_b=1e-5
% * momentum=0.5 (like in paper)
