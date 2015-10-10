% LBL_Module_NonLinearDynamics_bprop  Back-propagation, nonlinear dynamics
%
% Syntax:
%   [dL_dzHistory, model] = ...
%     LBL_Module_NonLinearDynamics_bprop(zHistory, zHidden, dL_dz, ...
%                                        thetas, model)
% Inputs:
%   zHistory: matrix of size <dim_z> x <n> x <n_samples> of latent history
%   zHidden:  matrix of size <dim_h> x <n_samples> of hidden activations
%   dL_dz:    matrix of size <dim_z> x <n_samples> of gradient
%             on the latent linear predictions
%   thetas:   matrix of topic mixture weights of size <dim_k> x <n_samples>
%   model:    struct containing the model
% Outputs:
%   dL_dzHistory: matrix of size <dim_z> x <n> x <n_samples> of gradients
%                 on the latent variables' history (n-gram)
%   model:        struct containing the model with updated gradients

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

function [dL_dzHistory, model] = ...
  LBL_Module_NonLinearDynamics_bprop(zHistory, zHidden, dL_dz, thetas, ...
  model)

% Number of predictions (samples) which were made
n_samples = size(dL_dz, 2);
n_topics = model.n_topics;

% Derivatives w.r.t. hidden activations
if (model.n_topics == 0)
  % Single topic (single dynamics)
  if ~isempty(model.B)
    dL_dzHidden = model.B' * dL_dz;
  else
    dL_dzHidden = dL_dz;
  end
else
  % Multiple topics/dynamics with mixture weights
  dL_dzHidden = cell(1, n_topics);
  for k = 1:n_topics
    coeffs = repmat(thetas(k, :), model.dim_h, 1);
    if ~isempty(model.B)
      dL_dzHidden{k} = coeffs .* (model.B{k}' * dL_dz);
    else
      dL_dzHidden{k} = coeffs .* dL_dz;
    end
  end
end

if ~isempty(model.B)
  % Derivative w.r.t. linear transformation matric B (second layer)
  if (model.n_topics == 0)
    % Single topic (single dynamics)
    dL_dB = dL_dz * zHidden';
  else
    % Multiple topics/dynamics with mixture weights
    dL_dB = cell(1, n_topics);
    for k = 1:n_topics
      coeffs = repmat(thetas(k, :), model.dim_zw, 1);
      dL_dB{k} = (coeffs .* dL_dz) * zHidden{k}';
    end
  end
  % Derivatives w.r.t. bias of the second layer
  if (n_topics == 0)
    % Single topic (single dynamics)
    dL_dbB = sum(dL_dz, 2);
  else
    % Multiple topics/dynamics with mixture weights
    dL_dbB = cell(1, n_topics);
    for k = 1:n_topics
      dL_dbB{k} = dL_dz * thetas(k, :)';
    end
  end
end

% Derivatives w.r.t. tanh nonlinearity
if (n_topics == 0)
  % Single topic (single dynamics)
  dL_dsum = dL_dzHidden .* (1 - zHidden.^2);
else
  % Multiple topics/dynamics with mixture weights
  dL_dsum = cell(1, n_topics);
  for k = 1:n_topics
    dL_dsum{k} = dL_dzHidden{k} .* (1 - zHidden{k}.^2);
  end
end

% Derivatives w.r.t. stacked histories of latent variables
if (model.n_topics == 0)
  % Single topic (single dynamics)
  % A' has size (<dim_z> * <n>) x <dim_h>
  % dL_dsum has size <dim_h> x <n_samples>
  % dL_dzHistory has size (<dim_z> * <n>) x <n_samples>
  dL_dzHistory = model.A' * dL_dsum;
else
  % Multiple topics/dynamics (but with no mixture weights here)
  dL_dzHistory = model.A{1}' * dL_dsum{1};
  for k = 2:model.n_topics
    dL_dzHistory = dL_dzHistory + model.A{k}' * dL_dsum{k};
  end
end
% Destack the history vectors into a sequence of <n> vectors
dL_dzHistory = reshape(dL_dzHistory, [model.dim_z, model.n, n_samples]);


if (nargout > 1)

  if ~isempty(model.B)
    % Store the gradients so far
    model.dL_dB = dL_dB;
    model.dL_dbB = dL_dbB;
  end
  
  % Derivative w.r.t. linear transformation matric A
  if (model.n_topics == 0)
    % Single topic (single dynamics)
    model.dL_dA = dL_dsum * reshape(zHistory, model.numel_zh, n_samples)';
  else
    % Multiple topics/dynamics (but with no mixture weights here)
    zHistory_reshaped = reshape(zHistory, model.numel_zh, n_samples)';
    for k = 1:model.n_topics
      model.dL_dA{k} = dL_dsum{k} * zHistory_reshaped;
    end
  end

  % Derivatives w.r.t. bias
  if (model.n_topics == 0)
    % Single topic (single dynamics)
    model.dL_dbA = sum(dL_dsum, 2);
  else
    % Multiple topics/dynamics (but with no mixture weights here)
    for k = 1:model.n_topics
      model.dL_dbA{k} = sum(dL_dsum{k}, 2);
    end
  end
end
