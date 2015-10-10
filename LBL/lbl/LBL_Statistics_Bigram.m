% LBL_Statistics_Bigram  Compute word co-occurence (bigram) statistics
%
% Syntax:
%   [dataset, graph] = LBL_Statistics_Bigram(dataset)
% Inputs:
%   dataset: struct containing the dataset
% Outputs:
%   dataset: struct containing the dataset
%   graph:   matrix of size <dim_w> x <dim_w> containing bigram statistics    

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

function [dataset, graph] = LBL_Statistics_Bigram(dataset)

% Use forward co-occurences
from = dataset.wHistories(:, end);
to = dataset.wTargets;
n_words = length(to);
dim_w = length(dataset.vocabulary);
graph = zeros(dim_w, dim_w);

% Loop over the words to get word counts
for k = 1:n_words
  from_k = from(k);
  to_k = to(k);
  graph(to_k, from_k) = graph(to_k, from_k) + 1;

  if (mod(k, 1000) == 0)
    fprintf(1, 'Computed bigram using %6d/%6d words\n', k, n_words);
  end
end

% Normalize the word counts along rows to get probabilities
for k = 1:dim_w
  norm_k = sum(graph(k, :));
  if (norm_k > 0)
    graph(k, :) = graph(k, :) / norm_k;
  end
end
dataset.bigram = graph;
