% NLP_ReadVocabulary  Slow way to read the vocabulary, if importdata fails
%
% Syntax:
%   vocabulary = NLP_ReadVocabulary(filename)
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

function vocabulary = NLP_ReadVocabulary(filename)

% Initialize the vocabulary
vocabulary = cell(100000, 1);
dim_w = 0;

% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens and parse the line
  while ~isempty(line)
    [tok, line] = strtok(line, ' ');
    if ~isempty(tok)
      k = 1;
      found = 0;
      while ((~found) && (k <= dim_w))
        if isequal(tok, vocabulary{k}), found = 1; else, k = k + 1; end
      end
      if ~found
        vocabulary{k} = tok;
        dim_w = k;
      end
    end
  end

  n_lines = n_lines + 1;
  if (mod(n_lines, 100) == 0)
    fprintf(1, 'Processed %5d lines\n', n_lines);
  end
end
fclose(fid);


% Find the beginning/end of sentence tags <S> and </S>
tag_begin = -1;
tag_end = -1;
tag_unk = -1;
for k = 1:dim_w
  if isequal(lower(vocabulary{k}), '<s>'), tag_begin = k; end
  if isequal(lower(vocabulary{k}), '</s>'), tag_end = k; end
  if isequal(lower(vocabulary{k}), '<unk>'), tag_unk = k; end
end
if (tag_begin <= 0)
  dim_w = dim_w + 1;
  tag_begin = dim_w;
  vocabulary{tag_begin} = '<S>';
end
if ((tag_end <= 0) && use_eos)
  dim_w = dim_w + 1;
  tag_end = dim_w;
  vocabulary{tag_end} = '</S>';
end
if ((tag_unk <= 0) && ~skip_oov)
  dim_w = dim_w + 1;
  tag_unk = dim_w;
  vocabulary{tag_unk} = '<UNK>';
end


% Sort the vocabulary alphabetically
vocabulary = vocabulary(1:dim_w);
vocabulary = sort(vocabulary);
