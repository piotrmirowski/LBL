function graph = NLP_BuildGraph(dataset, normalize, skipWords, trace)

if (nargin < 2)
  normalize = 0;
end
if (nargin < 3)
  skipWords = [];
end
if (nargin < 4)
  trace = 0;
end

n_samples = length(dataset.wTargets);
n_words = length(dataset.vocabulary);
graph = zeros(n_words, n_words);

for k = 1:n_samples
  graph(dataset.wTargets(k), dataset.wHistories(k, :)) = ...
    graph(dataset.wTargets(k), dataset.wHistories(k, :)) + 1;
  graph(dataset.wHistories(k, :), dataset.wTargets(k)) = ...
    graph(dataset.wHistories(k, :), dataset.wTargets(k)) + 1;
end

if ~isempty(skipWords)
  graph(skipWords, :) = 0;
  graph(:, skipWords) = 0;
end

if normalize
  for k = 1:n_words
    norm_k = sum(graph(k, :));
    if (norm_k > 0)
      graph(k, :) = graph(k, :) / norm_k;
    end
  end
end

if trace
  for k = 1:n_words
    fprintf(1, '%4d: %s ->', k, dataset.vocabulary{k});
    [dummy, ind_k] = sort(graph(k, :), 'descend');
    for i = 2:6
      fprintf(1, ' %s', dataset.vocabulary{ind_k(i)});
    end
    fprintf(1, '\n');
  end
end
