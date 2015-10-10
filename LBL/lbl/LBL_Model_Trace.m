% LBL_Model_Trace  Print the histories and targets, with predictions
%
% Syntax:
%   LBL_Model_Trace(wHistory, wTarget, P, model)
% Inputs:
%   wHistory: matrix of size <n> x <n_samples> word histories (indexes)
%   wTarget:  vector of length <n_samples> of word targets (indexes)
%   P:        matrix of size <dim_w> x <n_samples> of word probabilities
%   model:    struct containing the model

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

function LBL_Model_Trace(wHistory, wTarget, P, model)

n_samples = size(wHistory, 2);
for k = 1:n_samples
  for t = 1:model.n
    fprintf(1, '%s ', model.vocabulary{wHistory(t, k)});
  end
  [dummy, ind] = max(P(:, k));
  fprintf(1, '->%s (%s)\n', ...
    model.vocabulary{ind}, model.vocabulary{wTarget(k)});
end
