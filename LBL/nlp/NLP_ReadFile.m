function dataset = NLP_ReadFile(filename, n, vocabulary, skip_oov, use_eos)


% Vocabulary
vocabulary = sort(vocabulary);
dim_w = length(vocabulary);
dataset.vocabulary = vocabulary;
if (nargin < 4)
  skip_oov = 0;
end
if (nargin < 5)
  use_eos = 0;
end

% Find the beginning/end of sentence tags <S>/</S>, the unknown tag <UNK>
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


% Initialize the dataset structure
nHistory = n - 1;
dataset.wHistories = zeros(1000000, nHistory);
dataset.wTargets = zeros(1000000, 1);
n_nGrams = 0;
padding_begin = ones(1, nHistory) * tag_begin;
padding_end = ones(1, use_eos) * tag_end;


% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens and parse the line
  pos = 0;
  lineW = zeros(1, 1000);
  use_line = 1;
  while ~isempty(line)
    [tok, line] = strtok(line, ' ');
    if ~isempty(tok)
      k = 1;
      found = 0;
      while ((~found) && (k <= dim_w))
        if isequal(tok, vocabulary{k}), found = 1; else k = k + 1; end
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
      pos = pos + 1;
      lineW(pos) = k;
    end
  end
  if (use_line)
    lineW = [padding_begin lineW(1:pos) padding_end];

    % Get all n-gram histories and targets
    [histories, targets, m] = GetNGrams(lineW, nHistory);

    % Store them
    dataset.wHistories(n_nGrams + [1:m], :) = histories;
    dataset.wTargets(n_nGrams + [1:m]) = targets;
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


% -------------------------------------------------------------------------
function [histories, targets, m] = GetNGrams(vec, nHistory)

m = max(length(vec) - nHistory, 0);
histories = zeros(m, nHistory);
targets = zeros(m, 1);
for k = 1:m
  histories(k, :) = vec(k - 1 + [1:nHistory]);
  targets(k) = vec(k + nHistory);
end
