function dataset = ...
  NLP_ReadFileAPnewsWithPOS(filename, n, vocabulary, posTags)

% Vocabulary
dim_w = length(vocabulary);

% Find the beginning of sentence tag {bos}, and the unknown tag {unk}
tok_begin = '{bos}';
[tag_begin, found] = NLP_FindWord(tok_begin, vocabulary);
if ~found
  dim_w = dim_w + 1;
  tag_begin = dim_w;
  vocabulary{tag_begin} = tok_begin;
end
tok_unk = '<unknown>';
[tag_unk, found] = NLP_FindWord(tok_unk, vocabulary);
if ~found
  dim_w = dim_w + 1;
  tag_unk = dim_w;
  vocabulary{tag_unk} = tok_unk;
end


% Initialize the dataset structure
n_history = n - 1;
size_init = 15000000;
dataset.vocabulary = vocabulary;
dataset.tag_begin = tag_begin;
dataset.tag_unk = tag_unk;
dataset.wHistories = zeros(size_init, n_history);
dataset.wTargets = zeros(size_init, 1);
dataset.tagHistories = zeros(size_init, n_history);
dataset.tagTargets = zeros(size_init, 1);
dataset.documents = zeros(size_init, 1);
dataset.lines = zeros(size_init, 1);
n_nGrams = 0;

padding_begin = ones(1, n_history) * tag_begin;
padding_begin_tag = zeros(1, n_history);


% Open the file
fid = fopen(filename, 'r');

% Read the file line by line to extract (n+1)-grams
n_lines = 0;
n_documents = 1;
while 1
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Extract tokens, separate words from POS tags, and so parse the line
  pos = 0;
  lineW = zeros(1, 1000);
  lineT = zeros(1, 1000);
  use_line = 1;

  % Is it a new document?
  if isequal(line, './/_PERIOD_ ')
    n_documents = n_documents + 1;
  end
  
  % (continue) reading the line  
  while ~isempty(line)
    [tok, line] = strtok(line, ' ');
    if ~isempty(tok)

      % Identify the POS tag first
      [tok, tag] = strtok(tok, '/');
      tag = tag(3:end);
      [t, found] = NLP_FindWord(tag, posTags);
      if ~found
        t = 0;
        warning('Unknown POS tag "%s"', tag);
      end

      % Identify the lowercase word in the vocabulary
      tok = lower(tok);
      [w, found] = NLP_FindWord(tok, vocabulary);
      if ~found
        w = tag_unk;
        fprintf(1, 'Unknown word "%s"\n', tok);
      end
        
      % Record the word and POS tag and move on
      pos = pos + 1;
      lineW(pos) = w;
      lineT(pos) = t;
    end
  end
  
  if ((use_line) && (pos > 0))
    lineW = [padding_begin lineW(1:pos)];
    lineT = [padding_begin_tag lineT(1:pos)];

    % Get all n-gram histories and targets
    [histories, targets, m] = GetNGrams(lineW, n_history);
    [tagHistories, tagTargets] = GetNGrams(lineT, n_history);

    % Store them
    dataset.wHistories(n_nGrams + [1:m], :) = histories;
    dataset.wTargets(n_nGrams + [1:m]) = targets;
    dataset.tagHistories(n_nGrams + [1:m], :) = tagHistories;
    dataset.tagTargets(n_nGrams + [1:m]) = tagTargets;
    dataset.documents(n_nGrams + [1:m]) = n_documents;
    dataset.lines(n_nGrams + [1:m]) = n_lines;
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
dataset.documents = dataset.documents(1:n_nGrams);
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
