function dataset = ...
  NLP_ReadFileASR(filename, n, vocabulary, skip_oov, nbest)

% Vocabulary
dim_w = length(vocabulary);
if (nargin < 4)
  skip_oov = 0;
end
if (nargin < 5)
  nbest = 0;
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
dataset.tag_begin = tag_begin;
dataset.tag_unk = tag_unk;


% Initialize the dataset structure
n_history = n - 1;
size_init = 5000000;
dataset.wHistories = zeros(size_init, n_history);
dataset.wTargets = zeros(size_init, 1);
if (nbest)
  dataset.nBestRef = zeros(size_init, 1);
  dataset.nBestChoice = zeros(size_init, 1);
end
n_nGrams = 0;
padding_begin = ones(1, n_history) * tag_begin;


% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
n_nbest_ref = 0;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens and parse the line
  pos = 0;
  lineW = zeros(1, 1000);
  use_line = 1;

  % Special treatment for n-best
  if (nbest && ~isempty(line))
    [tok, line] = strtok(line, ' ');
    if isequal(tok, 'reference.')
      
      % Set a new n-best index
      n_nbest_choice = 0;
      n_nbest_ref = n_nbest_ref + 1;
    elseif isequal(tok, 'hypothesis.')
      n_nbest_choice = n_nbest_choice + 1;
    else
      error('n-best tag %s is neither "reference." or "hypothesis."', tok);
    end
  end

  % (continue) reading the line
  while ~isempty(line)
    [tok, line] = strtok(line, ' ');
    if ~isempty(tok)
      
      [k, found] = NLP_FindWord(tok, vocabulary);
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
      pos = pos + 1;
      lineW(pos) = k;
    end
  end
  if (use_line)
    lineW = [padding_begin lineW(1:pos)];

    % Get all n-gram histories and targets
    [histories, targets, m] = GetNGrams(lineW, n_history);

    % Store them
    dataset.wHistories(n_nGrams + [1:m], :) = histories;
    dataset.wTargets(n_nGrams + [1:m]) = targets;
    if (nbest)
      % Store the n-best indexes
      dataset.nBestRef(n_nGrams + [1:m]) = n_nbest_ref;
      dataset.nBestChoice(n_nGrams + [1:m]) = n_nbest_choice;
    end
    n_nGrams = n_nGrams + m;
    
    n_lines = n_lines + 1;
  end

  % Trace
  if (mod(n_lines, 100) == 0)
    fprintf(1, 'Processed %5d lines: %d n-grams\n', n_lines, n_nGrams);
  end
end
fclose(fid);


% Truncate the dataset
dataset.wHistories = dataset.wHistories(1:n_nGrams, :);
dataset.wTargets = dataset.wTargets(1:n_nGrams);
dataset.nBestRef = dataset.nBestRef(1:n_nGrams);
dataset.nBestChoice = dataset.nBestChoice(1:n_nGrams);


% -------------------------------------------------------------------------
function [histories, targets, m] = GetNGrams(vec, n_history)

m = max(length(vec) - n_history, 0);
histories = zeros(m, n_history);
targets = zeros(m, 1);
for k = 1:m
  histories(k, :) = vec(k - 1 + [1:n_history]);
  targets(k) = vec(k + n_history);
end
