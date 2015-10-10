% NLP_ReadFileWithPOS  Import text data with POS into a Matlab structure
%
% Syntax:
%   dataset = ...
%     NLP_ReadFileWithPOS(filename, n, vocabulary, posTags, [skip_oov])
% Inputs:
%   filename:   char array containing the text filename
%   n:          order of the n-gram (including prediction)
%   vocabulary: cell array with words, sorted
%   posTags:    cell array with POS tags, sorted
%   skip_oov:   boolean scalar: shall one skip sentences with OOV words?
%               (no by default)
% Outputs:
%   dataset:    struct containing the dataset

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

function dataset = ...
  NLP_ReadFileWithPOS(filename, n, vocabulary, posTags, skip_oov)


% Vocabulary
vocabulary = sort(vocabulary);
dim_w = length(vocabulary);

% Part-of-speech tags
dim_pos = length(posTags);
if (nargin < 5)
  skip_oov = 0;
end

% Find the beginning of sentence tag {bos}, and the unknown tag {unk}
tok_begin = '{bos}';
[tag_begin, found] = NLP_FindWord(tok_begin, vocabulary);
if ~found
  dim_w = dim_w + 1;
  tag_begin = dim_w;
  vocabulary{tag_begin} = tok_begin;
end
tok_unk = '{unk}';
[tag_unk, found] = NLP_FindWord(tok_unk, vocabulary);
if ~found
  dim_w = dim_w + 1;
  tag_unk = dim_w;
  vocabulary{tag_unk} = tok_unk;
end
dataset.vocabulary = vocabulary;


% Initialize the dataset structure
nHistory = n - 1;
dataset.wHistories = zeros(1000000, nHistory);
dataset.wTargets = zeros(1000000, 1);
dataset.tagHistories = zeros(1000000, nHistory);
dataset.tagTargets = zeros(1000000, 1);
n_nGrams = 0;
padding_begin = ones(1, nHistory) * tag_begin;
padding_begin_tag = zeros(1, nHistory);


% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens, separate words from POS tags, and so parse the line
  pos = 0;
  lineW = zeros(1, 1000);
  lineT = zeros(1, 1000);
  use_line = 1;
  while ~isempty(line)
    [tok, line] = strtok(line, ' ');
    if ~isempty(tok)

      % Identify the POS tag first
      [tok, tag] = strtok(tok, '/');
      tag = tag(3:end);
      t = 1;
      found = 0;
      while ((~found) && (t <= dim_pos))
        if isequal(tag, posTags{t}), found = 1; else t = t + 1; end
      end

      % Skip words that have no POS tag
      if found

        % Identify the word in the vocabulary
        w = 1;
        found = 0;
        while ((~found) && (w <= dim_w))
          if isequal(tok, vocabulary{w}), found = 1; else w = w + 1; end
        end
        if ~found
          if skip_oov

            % If cannot find the token, skip the line if asked to
            use_line = 0;
            fprintf(1, 'Skipping line containing %s...\n', tok);
          else
            k = tag_unk;
            fprintf(1, 'Unknown word %s\n', tok);
          end
        end
        
        % Record the word and POS tag and move on
        pos = pos + 1;
        lineW(pos) = w;
        lineT(pos) = t;
      end
    end
  end
  if (use_line)
    lineW = [padding_begin lineW(1:pos)];
    lineT = [padding_begin_tag lineT(1:pos)];

    % Get all n-gram histories and targets
    [histories, targets, m] = GetNGrams(lineW, nHistory);
    [tagHistories, tagTargets] = GetNGrams(lineT, nHistory);

    % Store them
    dataset.wHistories(n_nGrams + [1:m], :) = histories;
    dataset.wTargets(n_nGrams + [1:m]) = targets;
    dataset.tagHistories(n_nGrams + [1:m], :) = tagHistories;
    dataset.tagTargets(n_nGrams + [1:m]) = tagTargets;
    n_nGrams = n_nGrams + m;

    n_lines = n_lines + 1;
  end

  % Trace
  if (mod(n_lines, 100) == 0)
    fprintf(1, 'Processed %5d lines\n', n_lines);
  end
end
fclose(fid);


% Truncate the dataset
dataset.wHistories = dataset.wHistories(1:n_nGrams, :);
dataset.wTargets = dataset.wTargets(1:n_nGrams, :);
dataset.tagHistories = dataset.tagHistories(1:n_nGrams, :);
dataset.tagTargets = dataset.tagTargets(1:n_nGrams, :);


% -------------------------------------------------------------------------
function [histories, targets, m] = GetNGrams(vec, nHistory)

m = max(length(vec) - nHistory, 0);
histories = zeros(m, nHistory);
targets = zeros(m, 1);
for k = 1:m
  histories(k, :) = vec(k - 1 + [1:nHistory]);
  targets(k) = vec(k + nHistory);
end
