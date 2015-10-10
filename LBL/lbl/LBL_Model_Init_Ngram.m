% LBL_Model_Init_Ngram  Initialize word embedding from bigram statistics
%
% Syntax:
%   [model, dataset] = LBL_Model_Init_Ngram(model, dataset, n)
% Inputs:
%   model:   struct containing the model
%   dataset: struct containing the dataset
%   n:       order of the history
% Outputs:
%   model:   struct containing the model
%   dataset: struct containing the dataset

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

function [model, dataset] = LBL_Model_Init_Ngram(model, dataset, n)


% Recompute the bigram statistics if necessary
if ((n == 1) && ~isfield(dataset, 'bigram'))
  dataset = LBL_Statistics_Bigram(dataset);
  gram = dataset.bigram;
end
if ((n > 1) && ~isfield(dataset, 'ngram'))
  dataset = LBL_Statistics_Ngram(dataset, n);
  gram = dataset.ngram;
end

% Dimensionality reduction of the co-occurence bigram
fprintf(1, 'Computing %d-dimensional SVD for the word embedding\n', ...
  model.dim_zw);
[u, s, v] = svds(gram, model.dim_zw);

% Initial weights
model.R = (u * s)';

% Renormalize by the fan-in
std_r = std(model.R(:));
model.R = model.R * model.dim_w^(-1/2) / std_r;
