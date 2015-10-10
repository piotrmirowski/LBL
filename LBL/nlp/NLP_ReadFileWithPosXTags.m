% NLP_ReadFileWithPosXTags  Import text files with POS and Xtag into Matlab
%
% Syntax:
%   [dataset, vocabulary, posTags, superTags] = ...
%      NLP_ReadFileWithPosXTags(filename, n, vocabulary, ...
%                               posTags, superTags, [use_lower, ...
%                               skip_oov, use_eos, remove_cd, remove_nnp])
% Inputs:
%   filename:   character string with the filename
%   n:          scalar, order of the n-gram (history size is <n>-1)
%   vocabulary: cell array of |W| words, sorted alphabetically
%   posTags:    cell array of |X1| POS tags, sorted alphabetically
%   superTags:  cell array of |X2| super-tags, sorted alphabetically
%   use_lower:  boolean, indicating whether lower-case vocabulary is used
%   skip_oov:   boolean, indicating whether lines with out-of-vocabulary
%               words should be skipped (no by default)
%   use_eos:    boolean, indicating whether a special tag for 
%               end-of-sentence should be used (no by default)
%   remove_cd:  boolean, indicating whether numbers (POS tag CD)
%               should be removed (yes by default)
%   remove_nnp: boolean, indicating whether proper nouns (POS tags NNP(S))
%               should be removed (yes by default)
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

function [dataset, vocabulary, posTags, superTags] = ...
  NLP_ReadFileWithPosXTags(filename, n, vocabulary, ...
  posTags, superTags, use_lower, skip_oov, use_eos, remove_cd, remove_nnp)


% Vocabulary
dim_w = length(vocabulary);

% Find the beginning of sentence tag {bos}, the unknown tag {unk},
% the number tag {number} and the proper noun tag {proper}
tok_begin = '{bos}';
[tag_begin, found] = NLP_FindWord(tok_begin, vocabulary);
if ~found
  dim_w = dim_w + 1;
  tag_begin = dim_w;
  vocabulary{tag_begin} = tok_begin;
end
[tag_begin_pos, found] = NLP_FindWord('BOS', posTags);
if ~found
  tag_begin_pos = length(posTags) + 1;
  posTags{tag_begin_pos} = 'BOS';
end

% Remove numbers ("CD" POS tag)?
if (remove_cd)
  tok_cd = '{number}';
  [tag_cd, found] = NLP_FindWord(tok_cd, vocabulary);
  if ~found
    dim_w = dim_w + 1;
    tag_cd = dim_w;
    vocabulary{tag_cd} = tok_cd;
  end
else
  tag_cd = -1;
end

% Remove proper names ("NNP", "NNPS" POS tags)
if (remove_nnp)
  tok_nnp = '{proper}';
  [tag_nnp, found] = NLP_FindWord(tok_nnp, vocabulary);
  if ~found
    dim_w = dim_w + 1;
    tag_nnp = dim_w;
    vocabulary{tag_nnp} = tok_nnp;
  end
else
  tag_nnp = -1;
end

tok_unk = '<unknown>';
[tag_unk, found] = NLP_FindWord(tok_unk, vocabulary);
if ~found
  tok_unk = '{unk}';
  [tag_unk, found] = NLP_FindWord(tok_unk, vocabulary);
  if ~found
    dim_w = dim_w + 1;
    tag_unk = dim_w;
    vocabulary{tag_unk} = tok_unk;
  end
end
tok_end = '.';
[tag_end, found] = NLP_FindWord(tok_end, vocabulary);
if ~found, error('Could not find . (period) in the vocabulary'); end


% Find the CD and NNP, NNPS part-of-speech tags
if (remove_nnp)
  [pos_nnp, found] = NLP_FindWord('NNP', posTags);
  if ~found, error('The following POS tag was not found: NNP'); end
  [pos_nnps, found] = NLP_FindWord('NNPS', posTags);
  if ~found, error('The following POS tag was not found: NNPS'); end
end
% if ~found, pos_nnps = pos_nnp; end
if (remove_cd)
  [pos_cd, found] = NLP_FindWord('CD', posTags);
  if ~found, error('The following POS tag was not found: CD'); end
end

% Find the MISSING super-tag (for unknown ones)
[x_missing, found] = NLP_FindWord('MISSING', superTags);
if ~found, error('The following super-tag was not found: MISSING '); end


% Store the vocabulary and tags in the dataset
dataset.vocabulary = vocabulary;
dataset.posTags = posTags;
dataset.superTags = superTags;


% Default behaviour
if (nargin < 6)
  use_lower = 0;
end
if (nargin < 7)
  skip_oov = 0;
end
if (nargin < 8)
  use_eos = 0;
end
if (nargin < 9)
  remove_cd = 1;
end
if (nargin < 10)
  remove_nnp = 1;
end


% Initialize the dataset structure
len_wsj = 1500000;
n_history = n - 1;
dataset.wHistories = zeros(len_wsj, n_history);
dataset.wTargets = zeros(len_wsj, 1);
dataset.tagHistories = zeros(len_wsj, n_history);
dataset.posTagHistories = zeros(len_wsj, n_history);
dataset.xTagHistories = zeros(len_wsj, n_history);
dataset.sentences = zeros(len_wsj, 1);
dataset.lines = zeros(len_wsj, 1);

% n-gram counters
n_nGrams = 0;
lineW = zeros(1, 100);
lineP = zeros(1, 100);
lineS = zeros(1, 100);
use_line = 1;
k = 0;
padW = ones(1, n_history) * tag_begin;
padP = ones(1, n_history) * tag_begin_pos;
padS = zeros(1, n_history) * x_missing;

% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
l = 0;
n_sentences = 1;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens, separate words from POS tags, and so parse the line
  ind = strfind(line, '//');
  word = line(1:(ind-1));
  line = line((ind+2):end);
  % [word, line] = strtok(line, '//');
  ind = strfind(line, '//');
  posTok = line(1:(ind-1));
  superTok = line((ind+2):end);
  % [posTok, line] = strtok(line, '//');
  % superTok = line(3:end);

  if (use_lower)
    % Make lower case
    word = lower(word);
  end

  % Is it the end of a sentence?
  if ~isequal(lower(word), '...eos...')

    % Identify the word in the vocabulary
    [w, found] = NLP_FindWord(word, vocabulary, tag_begin, tag_end);
    if ~found
      if skip_oov
        % If cannot find the token, skip the line if asked to
        use_line = 0;
      else
        w = tag_unk;
        fprintf(1, 'Unknown word %s\n', word);
      end
    end
    % Identify the POS tag
    [p, found] = NLP_FindWord(posTok, posTags);
    if ~found, error('Unknown POS tag %s', posTok); end
    % Identify the super-tag
    [s, found] = NLP_FindWord(superTok, superTags);
    if ~found
      fprintf(1, 'Unknown xTag %s\n', superTok);
      s = x_missing;
    end
  
    % Replace the word tag by a generic one if its POS tag is NNP, NNPS, CD
    if (remove_cd && (p == pos_cd)) 
      w = tag_cd;
    elseif (remove_nnp && ((p == pos_nnp) || (p == pos_nnps)))
      w = tag_nnp;
    end

    % Record the word, POS tag, super-tag
    k = k + 1;
    lineW(k) = w;
    lineP(k) = p;
    lineS(k) = s;
  else

    % Only non-empty sentences that were not rejected are considered
    if ((k > 0) && use_line)

      % Remove EOS if required
      if (~use_eos && (lineW(k) == tag_end))
        k = k - 1;
      end

      % Pad the n-grams with {bos} tags
      try
        lineW = [padW lineW(1:k)];
        lineP = [padP lineP(1:k)];
        lineS = [padS lineS(1:k)];
      catch
        disp('toto');
      end

      % Get the n-grams
      [wHistories, wTargets, m_w] = GetNGrams(lineW, n_history);
      [pHistories, pTargets, m_p] = GetNGrams(lineP, n_history);
      [sHistories, sTargets, m_s] = GetNGrams(lineS, n_history);
      if ((m_w ~= m_p) || (m_w ~= m_s))
        error('Unequal number of elements');
      end

      % Store the n-grams
      dataset.wHistories(n_nGrams + (1:m_w), :) = wHistories;
      dataset.wTargets(n_nGrams + (1:m_w)) = wTargets;
      dataset.tagHistories(n_nGrams + (1:m_w), :) = sHistories;
      dataset.posTagHistories(n_nGrams + (1:m_w), :) = pHistories;
      dataset.xTagHistories(n_nGrams + (1:m_w), :) = sHistories;
      dataset.sentences(n_nGrams + (1:m_w)) = n_sentences;
      dataset.lines(n_nGrams + (1:m_w)) = n_sentences;
      n_nGrams = n_nGrams + m_w;

      % Jump to a new sentence
      n_sentences = n_sentences + 1;
      % Reset the token storage
      lineW = zeros(1, 100);
      lineP = zeros(1, 100);
      lineS = zeros(1, 100);
      k = 0;
      use_line = 1;
    end
  end

  % Trace
  if (mod(l, 1000) == 0)
    fprintf(1, 'Processed %7d words: %d n-grams, %d lines\n', ...
      l, n_nGrams, n_sentences - 1);
  end
  l = l + 1;
end
fclose(fid);


% Truncate the dataset
dataset.wHistories = dataset.wHistories(1:n_nGrams, :);
dataset.wTargets = dataset.wTargets(1:n_nGrams, :);
dataset.tagHistories = dataset.tagHistories(1:n_nGrams, :);
dataset.posTagHistories = dataset.posTagHistories(1:n_nGrams, :);
dataset.xTagHistories = dataset.xTagHistories(1:n_nGrams, :);
dataset.sentences = dataset.sentences(1:n_nGrams);
dataset.lines = dataset.lines(1:n_nGrams);


% -------------------------------------------------------------------------
function [histories, targets, m] = GetNGrams(vec, n_history)

m = max(length(vec) - n_history, 0);
histories = zeros(m, n_history);
targets = zeros(m, 1);
for k = 1:m
  histories(k, :) = vec(k - 1 + [1:n_history]);
  targets(k) = vec(k + n_history);
end
