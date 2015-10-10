% LBL_Dataset_Init  Prepare the order of the mini-batches in the dataset
%
% Syntax:
%   dataset = LBL_Dataset_Init(dataset, params, scramble)
% Inputs:
%   dataset:  struct containing the dataset used for training
%   params:   struct containing the parameters
%   scramble: boolean variable: shall we scramble the order of the batch
% Outputs:
%   dataset:  struct containing the ordered dataset used for training

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

function dataset = LBL_Dataset_Init(dataset, params, scramble)

% Samples
n_samples = length(dataset.wTargets);
dataset.n_samples = n_samples;

% Batches
dataset.n_batches = ceil(n_samples / params.len_batch);

% Scramble
if (scramble)
  dataset.order = randperm(n_samples);
else
  dataset.order = 1:n_samples;
end
