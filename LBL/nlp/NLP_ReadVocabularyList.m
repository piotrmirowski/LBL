% NLP_ReadVocabularyList  Read the vocabulary line by line
%
% Syntax:
%   vocabulary = NLP_ReadVocabularyList(filename)
% Inputs:
%   filename:   char array containing the text filename
% Outputs:
%   vocabulary: cell array with words, sorted

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
% Version 1.0, New York, 8 October 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu
%
% (c) 2010, AT&T Labs
%     180 Park Avenue, Florham Park, NJ 07932, USA.

function vocabulary = NLP_ReadVocabularyList(filename)

% Initialize the vocabulary
vocabulary = cell(100000, 1);

% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end
  n_lines = n_lines + 1;
  vocabulary{n_lines} = line;
  if (mod(n_lines, 100) == 0)
    fprintf(1, 'Processed %5d lines\n', n_lines);
  end
end
fclose(fid);

% Truncate and sort the list
vocabulary = sort(vocabulary(1:n_lines));
