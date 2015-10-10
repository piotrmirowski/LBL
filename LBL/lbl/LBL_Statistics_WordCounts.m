function wordCounts = ...
  LBL_Statistics_WordCounts(dataset, use_lines, window)


% Get the sequences of words, documents and lines
w = dataset.wTargets;

if (use_lines)
  d = dataset.lines;
else
  d = dataset.documents;
end
d = d - min(d) + 1;

% Number of words, documents and lines
dim_w = length(dataset.vocabulary);
n_words = length(w);
n_docs = max(d) - min(d) + 1;

% Word counts on the documents and lines
wordCounts = sparse(dim_w, n_docs);

% Compute the statistics by block
for k = 1:10000:n_words

  % Get matrix count for block
  fprintf(1, 'Block starting with %7d/%7d...\n', k, n_words);
  ind = k + (1:10000) - 1;
  ind = unique(min(ind, n_words));
  wordCounts_k = CountWords(w(ind), d(ind), dim_w, n_docs);

  % Add block
  wordCounts = wordCounts + wordCounts_k;
end

if ((nargin == 3) && (window > 1))
  % Sum over sliding window into past <window> "documents" (or lines)
  wordCountsWin = sparse(dim_w, n_docs);
  for k = 1:n_docs
    ind = unique(max((k-window+1):k, 1));
    wordCountsWin(:, k) = sum(wordCounts(:, ind), 2);
    if (mod(k, 1000) == 0)
      fprintf(1, 'Processed %7d/%7d documents...\n', k, n_docs);
    end
  end
  wordCounts = wordCountsWin;
end


% -------------------------------------------------------------------------
function wordCounts = CountWords(w, d, dim_w, n_docs)

n_words = length(w);

% Word counts on the documents and lines
wordCounts = sparse(dim_w, n_docs);
for k = 1:n_words
  wordCounts(w(k), d(k)) = wordCounts(w(k), d(k)) + 1;
  if (mod(k, 1000) == 0)
    fprintf(1, 'Processed %7d/%7d words...\n', k, n_words);
  end
end
