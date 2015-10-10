% IE_QuerySimilar  Get one or more samples using cosine similarity
%
% Syntax:
%   [indAnswers, similarity] = ...
%     IE_QuerySimilar(z, q, n_answers, [normZ, normQ])
% Inputs:
%   z:          matrix of size <M> x <T> with a document representation
%   q:          vector of size <M> x 1 with a query document representation
%   n_answers:  number of desired answers (best matches)
%   normZ:      vector of size 1 x <T> of norms for the documents;
%               it is recomputed if necessary
%   normQ:      norm of the query document; recomputed if necessary
% Outputs:
%   indAnswers: indexes of best matches
%   similarity: score, going from max (best) to min (worst)

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
% Version 1.0, New York, 12 March 2010
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [indAnswers, similarity] = ...
  IE_QuerySimilar(z, q, n_answers, normZ, normQ)

if (nargin < 4)
  normZ = full(sqrt(sum(z.^2)));
end
if (nargin < 5)
  normQ = full(sqrt(sum(q.^2)));
end

% Compute cosine distance between data and query
similarity = full(z' * q)' ./ (normZ * normQ);
[similarity, orderQuery] = sort(similarity, 'descend');

% Return the indexes of the top <n_answers>
indAnswers = orderQuery(1:n_answers);
similarity = similarity(1:n_answers);

