% LBL_Evaluate_Embedding  Print top <n> closest words in embedding space
%
% Syntax:
%   LBL_Evaluate_Embedding(model, fid, indexes, n)
% Inputs:
%   model:   struct containing the model being evaluated
%   fid:     scalar containing the file identifier for output (1 = screen)
%   indexes: optional vector of word indexes to be evaluated (default: all)
%   n:       optional number of word to print (default: 7)

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

function LBL_Evaluate_Embedding(model, fid, indexes, n)

% Word embedding
n_words = size(model.R, 2);
R = model.R;
normR = full(sqrt(sum(R.^2)));

if (nargin < 3)
  indexes = 1:n_words;
end
if (nargin < 4)
  n = 7;
end

% Loop over specified indexes
for k = indexes
  Rk = R(:, k);
  indTopSimilar = IE_QuerySimilar(R, Rk, n, normR, normR(k));

  fprintf(fid, '%4d: %s ->', k, model.vocabulary{k});
  for i = 1:n
    fprintf(fid, ' %s', model.vocabulary{indTopSimilar(i)});
  end
  fprintf(fid, '\n');
end

