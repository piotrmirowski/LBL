% IE_ComputeTFIDF
%   Compute the term frequency–inverse document frequency (TFIDF), as well
%   as the term frequency (TF) and document frequency (DF) of word counts.
%
% Syntax:
%   [tfidf, tf, df] = IE_ComputeTFIDF(wordCounts[, idf])
%
% Inputs:
%   wordCounts: (possibly sparse) matrix of word counts of size
%               <nWords> x <nDocs>
%   idf:        optional vector of inverse DF of size <nWords> x 1
%
% Returns:
%   tfidf:      sparse matrix of TFIDF of size <nWords> x <nDocs>
%   tf:         sparse matrix of TF of size <nWords> x <nDocs>
%   df:         sparse vector of DF of size <nWords> x 1

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
% Version 1.0, New York, 18 July 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [tfidf, idf, tf, df] = IE_ComputeTFIDF(wordCounts, idf)

nDocs = size(wordCounts, 2);
nWords = size(wordCounts, 1);

if nargin < 2
  % Recompute the inverse document frequency or use a standardized one
  fprintf(1, 'Recomputing the word document frequency...\n');
  df = sum(wordCounts > 0, 2);
  idf = log(nDocs ./ df);
end

tf = sparse(nWords, nDocs);
tfidf = sparse(nWords, nDocs);
for j = 1:nDocs
  ind = find(wordCounts(:, j));
  tf_j = wordCounts(ind, j) / sum(wordCounts(ind, j));
  tf(ind, j) = tf_j;
  tfidf(ind, j) = tf_j .* idf(ind);
  
  % Trace
  if (mod(j, 100) == 0)
    fprintf(1, 'Computed tfidf on %d/%d documents...\n', j, nDocs);
  end
end
