function NLP_WriteDataset(dataset, filename)

n_lines = dataset.lines(end);
vocabulary = dataset.vocabulary;
wTargets = dataset.wTargets;

% Write all the dataset, one line per sentence
fid = fopen(filename, 'w');
for k = 1:n_lines
  ind_k = find(dataset.lines == k);
  n_k = length(ind_k);
  if (n_k > 0)
    fprintf(fid, '%s', vocabulary{wTargets(ind_k(1))});
    for j = 2:n_k
      fprintf(fid, ' %s', vocabulary{wTargets(ind_k(j))});
    end
    fprintf(fid, '\n');
  end
  
  if (mod(k, 100) == 0)
    fprintf(1, 'Wrote %5d/%5d lines\n', k, n_lines);
  end
end
fclose(fid);
