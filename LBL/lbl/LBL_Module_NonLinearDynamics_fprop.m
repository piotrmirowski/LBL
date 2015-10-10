% LBL_Module_NonLinearDynamics_fprop  Predict latent vars with nonlinearity
%
% Syntax:
%   [zNonLinear, zHidden] = ...
%     LBL_Module_NonLinearDynamics_fprop(zHistory, thetas, model)
% Inputs:
%   zHistory: matrix of size <dim_z> x <n> x <n_samples> of latent history
%   thetas:   matrix of topic mixture weights of size <dim_k> x <n_samples>
%   model:    struct containing the model
% Outputs:
%   zLinear:  matrix of size <dim_z> x <n_samples> of latent predictions
%   zHidden:  matrix of size <dim_h> x <n_samples> of hidden activations,
%             or cell array of <dim_k> elements of such matrices

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

function [zNonLinear, zHidden] = ...
  LBL_Module_NonLinearDynamics_fprop(zHistory, thetas, model)

% Number of predictions (samples) which were made
n_samples = size(zHistory, 3);
n_topics = model.n_topics;

% Use a stacked history and single A matrix to produce hidden activations
if (n_topics == 0)
  % Single topic (single dynamics)
  zHidden = model.A * reshape(zHistory, model.numel_zh, n_samples) + ...
    repmat(model.Abias, 1, n_samples);
else
  % Multiple topics/dynamics (do not use mixture weights yet)
  zHidden = cell(1, n_topics);
  zHistory_reshape = reshape(zHistory, model.numel_zh, n_samples);
  for k = 1:n_topics
    zHidden{k} = ...
      model.A{k} * zHistory_reshape + repmat(model.Abias{k}, 1, n_samples);
  end
end

% Nonlinearity
if (n_topics == 0)
  % Single topic (single dynamics)
  zHidden = tanh(zHidden);
else
  % Multiple topics/dynamics with mixture weights
  for k = 1:n_topics
    zHidden{k} = tanh(zHidden{k});
  end
end

% Produce the prediction
if (n_topics == 0)
  % Single topic (single dynamics)
  if ~isempty(model.B)
    zNonLinear = model.B * zHidden + repmat(model.Bbias, 1, n_samples);
  else
    zNonLinear = zHidden;
  end
else
  % Multiple topics/dynamics (use mixture weights here)
  coeffs = repmat(thetas(1, :), model.dim_zw, 1);
  if ~isempty(model.B)
    zNonLinear = coeffs .* ...
      (model.B{1} * zHidden{1} + repmat(model.Bbias{1}, 1, n_samples));
  else
    zNonLinear = coeffs .* zHidden{1};
  end
  for k = 2:n_topics
    coeffs = repmat(thetas(k, :), model.dim_zw, 1);
    if ~isempty(model.B)
      zNonLinear = zNonLinear + coeffs .* ...
        (model.B{k} * zHidden{k} + repmat(model.Bbias{k}, 1, n_samples));
    else
      zNonLinear = zNonLinear + coeffs .* zHidden{k};
    end
  end
end
