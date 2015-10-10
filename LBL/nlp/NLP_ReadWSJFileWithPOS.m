% NLP_ReadWSJFileWithPOS  Import the WSJ text files into a Matlab structure
%
% Syntax:
%   dataset = NLP_ReadWSJFileWithPOS(filename, n, vocabulary, ...
%                                    posTags, [skip_oov, ...
%                                    use_eos, remove_cd, remove_nnp])
% Inputs:
%   filename:   character string with the filename
%   n:          scalar, order of the n-gram (history size is <n>-1)
%   vocabulary: cell array of |W| words, sorted alphabetically
%   posTags:    cell array of |X| POS tags, sorted alphabetically
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

function dataset = NLP_ReadWSJFileWithPOS(filename, n, vocabulary, ...
  posTags, skip_oov, use_eos, remove_cd, remove_nnp)


% Vocabulary
dataset.vocabulary = vocabulary;

% Part-of-speech tags
if (nargin < 5)
  skip_oov = 0;
end
if (nargin < 6)
  use_eos = 0;
end
if (nargin < 7)
  remove_cd = 1;
end
if (nargin < 8)
  remove_nnp = 1;
end

% Find the beginning/end of sentence, number, proper name and unknown tags
[tag_begin, found] = NLP_FindWord('bos', vocabulary);
if ~found, error('No bos'); end
[tag_end, found] = NLP_FindWord('_period_', vocabulary);
if ~found, error('No _period_'); end
[tag_unk, found] = NLP_FindWord('<unk>', vocabulary);
if (~found && ~skip_oov), error('No <unk>'); end
[tag_cd, found] = NLP_FindWord('_number_', vocabulary);
if ~found, error('No _number_'); end
[tag_nnp, found] = NLP_FindWord('_proper_', vocabulary);
if ~found, error('No _proper_'); end

% Find the CD and NNP, NNPS part-of-speech tags
[pos_nnp, found] = NLP_FindWord('NNP', posTags);
if ~found, error('No NNP'); end
[pos_nnps, found] = NLP_FindWord('NNPS', posTags);
if ~found, error('No NNPS'); end
[pos_cd, found] = NLP_FindWord('CD', posTags);
if ~found, error('No CD'); end


% Initialize the dataset structure
nHistory = n - 1;
dataset.wHistories = zeros(1200000, nHistory);
dataset.wTargets = zeros(1200000, 1);
dataset.tagHistories = zeros(1200000, nHistory);
dataset.tagTargets = zeros(1200000, 1);
dataset.sentences = zeros(1200000, 1);
dataset.lines = zeros(1200000, 1);
n_nGrams = 0;


% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
n_sentences = 1;
just_skipped = 0;
while 1
  line = fgetl(fid);
  line0 = line;
  if ~ischar(line), break; end

  % Extract tokens, separate words from POS tags, and so parse the line
  p = 0;
  lineW = zeros(1, 100);
  lineT = zeros(1, 100);
  use_line = 1;
  recent_oov = 0;
  while ~isempty(line)
    [tok, line] = strtok(line, ',');
    if ~isempty(tok)
      p = p + 1;

      % Is it a POS tag?
      if (mod(p, 2) == 0)
        pos = tok;

        % Identify the POS tag
        [t, found] = NLP_FindWord(pos, posTags);
        if ~found
          error('Unknown POS: %s', pos);
        end

        % Replace previously found word tag by generic one if its POS tag
        % is NNP, NNPS or CD
        if ((t == pos_cd) && remove_cd)
          lineW(p/2) = tag_cd;
        elseif ((t == pos_nnp) && remove_nnp)
          lineW(p/2) = tag_nnp;
        elseif ((t == pos_nnps) && remove_nnp)
          lineW(p/2) = tag_nnp;
        end
        
        % Reverse a decision not to use a line in case of OOV numbers or
        % proper nouns
        if (~use_line && recent_oov)
          if (((t == pos_cd) && remove_cd) || ...
              (((t == pos_nnp) || (t == pos_nnps)) && remove_nnp))
            use_line = 1;
          else
            fprintf(1, '# skip unknown word %s(%s)\n', word, pos); 
          end
        end
        
        % Record the POS tag and move on
        lineT(p/2) = t;
      else
        word = tok;
        recent_oov = 0;
        
        % Make lower case
        word = lower(word);
      
        % Identify the word in the vocabulary
        [w, found] = NLP_FindWord(word, vocabulary, tag_begin, tag_end);
        if ~found
          if use_line
            recent_oov = 1;
          end
          if skip_oov

            % If cannot find the token, skip the line if asked to
            use_line = 0;
          else
            w = tag_unk;
            fprintf(1, 'Unknown word %s\n', word);
          end
        end

        % Record the word tag and move on
        lineW((p+1)/2) = w;
      end
    end
  end
  
  % Have we read the same number of words and POS tags?
  if (mod(p, 2) ~= 0)
    error('Incorrect # word and POS on line %s', line0);
  end
  p = p / 2;

  % Are we predicting the end of the sentence?
  if (w == tag_end)
    if ~just_skipped
      n_sentences = n_sentences + 1;
    end
    if (use_line && ~use_eos)
      use_line = 0;
      fprintf(1, '. skip end of sentence\n');
      just_skipped = 1;
    end
  end

  n_lines = n_lines + 1;
  if (use_line)
    lineW = lineW(p + [(-n+1):0]);
    lineT = lineT(p + [(-n+1):0]);

    % Store them
    n_nGrams = n_nGrams + 1;
    dataset.wHistories(n_nGrams, :) = lineW(1:nHistory);
    dataset.wTargets(n_nGrams) = lineW(n);
    dataset.tagHistories(n_nGrams, :) = lineT(1:nHistory);
    dataset.tagTargets(n_nGrams) = lineT(n);
    dataset.sentences(n_nGrams) = n_sentences;
    dataset.lines(n_nGrams) = n_lines;
    just_skipped = 0;
  else

    % Increase the number of sentences (unless it is a stream of unknown)
    if ~just_skipped
      n_sentences = n_sentences + 1;
    end
    fprintf(1, ' -> skipping %6d: %s...\n', n_lines, line0);
    just_skipped = 1;
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
dataset.sentences = dataset.sentences(1:n_nGrams);
dataset.lines = dataset.lines(1:n_nGrams);
