% NLP_FindWord  Fast search of a word token in a sorted list
%
% Syntax:
%   [w, found] = NLP_FindWord(tok, vocabulary, tag_1, tag_2)
% Inputs:
%   tok:        character string with the token to find
%   vocabulary: cell array of N word tokens, sorted alphabetically
%   tag_1:      position in <vocabulary> of token #1 (for fast retrieval)
%   tag_2:      position in <vocabulary> of token #2 (for fast retrieval)
% Outputs:
%   w:          position in <vocabulary> of retrieved token, if found
%   found:      boolean indicating whether <tok> was found in <vocabulary> 

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

function [w, found, k] = NLP_FindWord(tok, vocabulary, tag_1, tag_2)

k = 0;

if ((nargin > 2) && isequal(tok, vocabulary{tag_1}))
  w = tag_1;
  found = 1;
  return;
end
if ((nargin > 3) && isequal(tok, vocabulary{tag_2}))
  w = tag_2;
  found = 1;
  return;
end

% Identify the word in the vocabulary
w = 1;
found = 0;
a = 1;
b = length(vocabulary);

if isequal(tok, vocabulary{a})
  w = a;
  found = 1;
  return;
end
if isequal(tok, vocabulary{b})
  w = b;
  found = 1;
  return;
end

while ((~found) && (a < b))
  w = floor((a + b) / 2);
  res = strlexcmp(tok, vocabulary{w});
  if (res == 0)
    found = 1;
    return;
  elseif (res < 0)
    b = w;
  elseif (a == w)
    w = -1;
    return;
  else
    a = w;
  end
  k = k + 1;
end

if (~found)
  w = -1;
end
