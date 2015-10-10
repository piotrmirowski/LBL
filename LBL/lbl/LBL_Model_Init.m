% LBL_Model_Init  Create a modular language model
%
% Syntax:
%   model = LBL_Model_Init(params, dataset)
% Inputs:
%   params:  struct containing the parameters
%   dataset: struct containing the dataset (we need the word unigram proba
%            to initialize the biases of the word embedding)
% Outputs:
%   model:   struct containing the model

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

function model = LBL_Model_Init(params, dataset)


fprintf(1, 'Initializing the model...\n');


% Dimensions of the modules
dim_w = params.dim_w;
dim_x = params.dim_x;
dim_z = params.dim_z;
dim_zw = params.dim_zw;
dim_zx = params.dim_zx;
n_dim_z = params.n * dim_z;
dim_h = params.dim_h;
n_topics = params.n_topics;

% Sanity check
if (((dim_zw < dim_z) && (dim_x == 0)) || ...
    ((dim_zx == 0) && (dim_x > 0)))
  error('Inconsistent dimensions for the input feature layer');
end
if ((dim_zw + dim_zx) ~= dim_z)
  error('Inconsistent # dimensions for the latent variables');
end


% Basic components of the model
model = struct('n', params.n, ...
  'dim_w', dim_w, 'dim_x', dim_x, ...
  'dim_z', dim_z, 'dim_zw', dim_zw, 'dim_zx', dim_zx, ...
  'numel_zh', n_dim_z, 'dim_h', dim_h, 'n_topics', n_topics, ...
  'measure', inf, ...
  'R', [], 'bw', [], ...                          % Word embedding
  'C', [], 'Cbias', [], ...                       % Linear dynamics
  'A', [], 'Abias', [], 'B', [], 'Bbias', [], ... % Nonlinear dynamics
  'F', [], 'Fbias', [], ...                       % Feature embedding
  'graphNeighb', [], 'graphSamples', []);


% Different random weight initializations for the word embedding
switch (params.init_r)
  case 'inv' % bad
    model.R = randn(dim_zw, dim_w) / dim_w;
  case 'inv_sqrt' % this is the only good one
    model.R = randn(dim_zw, dim_w) * dim_w^(-1/2);
  case 'unit' % bad
    model.R = randn(dim_zw, dim_w);
  case 'bigram'
    model = LBL_Model_Init_Ngram(model, dataset, 1);
  case 'ngram'
    model = LBL_Model_Init_Ngram(model, dataset, params.n);
  case '0and1'
    model.R = randn(dim_zw, dim_w) * dim_w^(-1/2);
    model.R = tanh(model.R);
end
model.bw = zeros(dim_w, 1);

% Linear dynamics module
if (n_topics == 0)
  % Single dynamics
  model.C = randn(dim_zw, n_dim_z) * n_dim_z^(-1/2);
  model.Cbias = zeros(dim_zw, 1);
else
  % Mixture
  model.C = cell(1, n_topics);
  model.Cbias = cell(1, n_topics);
  for k = 1:n_topics
    model.C{k} = randn(dim_zw, n_dim_z) * n_dim_z^(-1/2);
    model.Cbias{k} = zeros(dim_zw, 1);
  end
end

% Vocabulary and unigrams (for initialization of biases)
model.vocabulary = dataset.vocabulary;
dataset = LBL_Statistics_Unigram(dataset);
model.bw = dataset.unigram / max(dataset.unigram);

% Set to zero the unknown representation
if (params.open_vocab)
  fprintf(1, 'Using an open-vocabulary model with <unk>...\n');
  model.tag_unk = dataset.tag_unk;
  model.R(:, model.tag_unk) = 0;
end

% Nonlinear component
if (dim_h > 0)
  if (n_topics == 0)
    % Single dynamics
    model.A = randn(dim_h, n_dim_z) * n_dim_z^(-1/2);
    model.Abias = zeros(dim_h, 1);
    if (params.use_neu2)
      model.B = randn(dim_zw, dim_h) * dim_h^(-1/2);
      model.Bbias = zeros(dim_zw, 1);
    end
  else
    % Mixture
    model.A = cell(1, n_topics);
    model.Abias = cell(1, n_topics);
    if (params.use_neu2)
      model.B = cell(1, n_topics);
      model.Bbias = cell(1, n_topics);
    end
    for k = 1:n_topics
      model.A{k} = randn(dim_h, n_dim_z) * n_dim_z^(-1/2);
      model.Abias{k} = zeros(dim_h, 1);
      if (params.use_neu2)
        model.B{k} = randn(dim_zw, dim_h) * dim_h^(-1/2);
        model.Bbias{k} = zeros(dim_zw, 1);
      end
    end
  end
end

% Feature embedding
if (dim_x > 0)
  if (dim_x ~= model.dim_zx)
    model.F = randn(dim_zx, dim_x) * dim_x^(-1/2);
  else
    model.F = eye(dim_x);
  end
  model.Fbias = zeros(dim_zx, 1);
end


% Log-likelihood sampling: initialize the neighborhood and samples graphs
if (params.approx_ll)
  model = LBL_LogLikelihood_BiLinearApprox_Init(model, params);
end
