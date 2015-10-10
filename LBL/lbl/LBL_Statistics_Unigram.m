% LBL_Statistics_Unigram  Compute per word (unigram) statistics
%
% Syntax:
%   [dataset, p] = LBL_Statistics_Unigram(dataset)
% Inputs:
%   dataset: struct containing the dataset
% Outputs:
%   dataset: struct containing the dataset
%   p:       vector of length <dim_w> containing unigram statistics    

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

function [dataset, p] = LBL_Statistics_Unigram(dataset)

dim_w = length(dataset.vocabulary);
if (~isfield(dataset, 'unigram') || (length(dataset.unigram) ~= dim_w))

  % Compute the unigram on the target words
  fprintf(1, 'Computing the unigram on the %d-word dataset...\n', dim_w);
  p = zeros(dim_w, 1);
  for k = 1:dim_w
    p(k) = sum(dataset.wTargets == k);
    if (mod(k, 1000) == 0)
      fprintf(1, 'o');
    elseif (mod(k, 100) == 0)
      fprintf(1, '.');
    end
  end
  fprintf(1, '\n');
  p = p / sum(p);
  dataset.unigram = p;
end
