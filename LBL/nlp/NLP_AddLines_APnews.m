function dataset = NLP_AddLines_APnews(dataset)

% Find all the periods (beginning lines)
tag_period = dataset.tag_period;
indPeriod = find(dataset.wTargets == tag_period);
n_periods = length(indPeriod);

n_samples = length(dataset.wTargets);
lines = zeros(n_samples, 1);
documents = zeros(n_samples, 1);

% Loop over the periods
i_last = 1;
i_begin = 1;
n_lines = 0;
n_docs = 0;
for k = 1:n_periods
  i = indPeriod(k);
  if (i == i_last)

    % New document
    n_docs = n_docs + 1;
    documents(i_begin:i) = n_docs;
    i_begin = i + 1;
  else
    
    % New line
    n_lines = n_lines + 1;
    a = i_last;
    b = i;
    lines(a:b) = n_lines;
  end
  i_last = i + 1;
  
  if (mod(k, 1000) == 0)
    fprintf(1, 'Processed %d/%d periods...\n', k, n_periods);
  end
end

% Last line
if (i < n_samples)
  lines((i+1):end) = n_lines + 1;
end

% Last document
if (i_begin < n_samples)
  documents(i_begin:end) = n_docs + 1;
end

% Store the lines and documents
dataset.lines = lines;
dataset.documents = documents;
