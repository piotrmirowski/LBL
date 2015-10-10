function LBL_Trainer_Batch(datasetTrain, datasetXval, params, k0, varargin)

% Retrieve the parameter names and their ranges
n_params = floor((nargin - 3) / 2);
argNames = cell(1, n_params);
argValues = cell(1, n_params);
nValues = zeros(1, n_params);
for k = 1:n_params
  argNames{k} = varargin{2*k-1};
  argValues{k} = varargin{2*k};
  nValues(k) = length(argValues{k});
end
a = zeros(1, n_params);
b = zeros(1, n_params);

% Loop over all the combinations of parameter values
n_values_total = prod(nValues);
for k = k0:n_values_total
  i = k;
  for j = 1:n_params
    a(j) = mod(i - 1, nValues(j));
    i = (i - a(j) - 1) / nValues(j) + 1;
    b(j) = a(j) + 1;
  end

  % Set the parameter modificators
  for j = 1:n_params
    arg_name = argNames{j};
    arg_val = argValues{j}(b(j));
    eval(['params.' arg_name ' = ' num2str(arg_val) ';']);
  end

  % Modify the batch name
  % Filename
  params.filename = ...
    sprintf('w%d_n%d_z%d_h%d_x%d_%s_b%3d', params.dim_w, params.n, ...
    params.dim_z, params.dim_h, params.dim_x, ...
    datestr(now, 'yyyymmdd_HHMMss'), k);
  params.filename = strrep(params.filename, ' ', '0');
  params.batch = k;
  
  % Trace
  fprintf(1, 'Launching batch process %d/%d with following params:\n', ...
    k, n_values_total);
  disp(params);
  
  % Launch the training session with the modified parameters
  LBL_Trainer(datasetTrain, datasetXval, params);
end
