% LBL_Dataset_fprop  Get the <k>-th mini-batch from the dataset
%
% Syntax:
%   [wTarget, wHistory, featTarget, featHistory, thetas, indK] = ...
%     LBL_Dataset_fprop(dataset, params, k)
% Inputs:
%   dataset: struct containing the dataset used for training/evaluation
%   params:  struct containing the parameters
%   k:       scalar, number of the mini-batch to be retrieved
% Outputs:
%   wTarget:     vector of size 1 x <n_samples> of target word indexes
%   featTarget:  matrix of size <dim_x> x <n_samples> of target features
%   wHistory:    matrix of size <n> x <n_samples> of n-gram word indexes
%   featHistory: matrix of size <dim_x> x <n> x <n_samples> of features
%   thetas:      matrix of topic mixtures of size <dim_k> x <n_samples>
%   indK:        vector of length <n_samples> of sample indexes

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
%
% (c) 2010, AT&T Labs
%     180 Park Avenue, Florham Park, NJ 07932, USA.

function [wTarget, wHistory, xHistory, thetas, indK] = ...
  LBL_Dataset_fprop(dataset, params, k)

len_batch = params.len_batch;
n_samples = length(dataset.wTargets);
k0 = (k - 1) * len_batch + 1;
k1 = k * len_batch;
indK = unique(min([k0:k1], n_samples));

% Try using a scrambled order of datapoints
try
  indK = dataset.order(indK);
end

% Select the minibatch of words (word indexes)
wTarget = dataset.wTargets(indK)';
wHistory = dataset.wHistories(indK, :)';

% Word features (POS, super-tags, etc...)
if (params.dim_x > 0)
  % Select the minibatch of word features
  n_tags = params.dim_x; % Currently, only discrete POS are features
  len_batch = length(indK);
  dimFeat = [n_tags, len_batch];
  xHistory = zeros(n_tags, params.n, len_batch);
  for i = 1:params.n
    tagsI = dataset.tagHistories(indK, i)';
    indI = find(tagsI);
    indHistory_i = sub2ind(dimFeat, tagsI(indI), indI);
    xHistory_i = zeros(dimFeat);
    xHistory_i(indHistory_i) = 1;
    xHistory(:, i, :) = xHistory_i;
  end
else
  % No features
  xHistory = [];
end

% Document/line features (topics, etc...)
if (params.n_topics > 0)
  % Select the topics
  thetas = dataset.topics(indK, :)';
else
  % No topics
  thetas = [];
end
