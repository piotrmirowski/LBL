% NLP_ReMapWords  Given new vocabulary index map, remap history and targets
%
% Syntax:
%   dataset = NLP_ReMapWords(dataset, indMapWords, tag)
% Inputs:
%   dataset:     struct containing the dataset
%   indMapWords: vector of length <dim_w>, showing where each word goes
%   tag:         scalar index of "rare" words in the vocabulary
% Outputs:
%   dataset:     struct containing the dataset

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

function dataset = NLP_ReMapWords(dataset, indMapWords, tag)

dim_w_reduced = length(indMapWords);
if (nargin < 3)
  tag = 0;
end

% Loop over the entire vocabulary to deplace the word indexes
for j = 1:dim_w_reduced
  k = indMapWords(j);
  indTargets = (dataset.wTargets == k);
  indHistories = (dataset.wHistories == k);
  val_j = tag + ~tag * j;
  dataset.wTargets(indTargets) = val_j;
  dataset.wHistories(indHistories) = val_j;
  
  if (mod(j, 100) == 0)
    fprintf(1, 'Remapped %5d/%5d words...\n', j, dim_w_reduced);
  end
end
