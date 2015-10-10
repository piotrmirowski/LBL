% LBL_Statistics_Ngram  Compute word co-occurence (n-gram) statistics
%
% Syntax:
%   [dataset, graph] = LBL_Statistics_Ngram(dataset, n)
% Inputs:
%   dataset: struct containing the dataset
%   n:       order of the n-gram
% Outputs:
%   dataset: struct containing the dataset
%   graph:   matrix of size <dim_w> x <dim_wn> containing n-gram statistics
%            where <dim_w> = <dim_w> * <n>

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

function [dataset, graph] = LBL_Statistics_Ngram(dataset, n)


% Trivial cases
if (n == 1)
  [dataset, graph] = LBL_Statistics_Bigram(dataset);
  return;
end
if ((n < 1) || (n > size(dataset.wHistories, 2)))
  error('<n> needs to be at least 1 and less than the dataset''s order');
end


% Use forward co-occurences
from = dataset.wHistories(:, n:(-1):1);
to = dataset.wTargets;
n_words = length(to);
dim_w = length(dataset.vocabulary);
graph = zeros(dim_w, dim_w * n);

% Loop over the words to get word counts
for i = 1:n
  offset_i = (i - 1) * dim_w;
  for k = 1:n_words
    from_k = from(k, i);
    from_ki = from_k + offset_i;
    to_k = to(k);
    graph(to_k, from_ki) = graph(to_k, from_ki) + 1;

    if (mod(k, 1000) == 0)
      fprintf(1, 'Computed bigram %d/%d using %6d/%6d words\n', ...
        i, n, k, n_words);
    end
  end
end

% Normalize the word counts along rows to get probabilities
for i = 1:n
  indI = (1:dim_w) + (i - 1) * dim_w;
  for k = 1:dim_w
    norm_k = sum(graph(k, indI));
    if (norm_k > 0)
      graph(k, indI) = graph(k, indI) / norm_k;
    end
  end
end
dataset.ngram = graph;
