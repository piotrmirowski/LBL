% LBL_Module_LinearDynamics_bprop  Back-propagation for linear dynamics
%
% Syntax:
%   [dL_dzHistory, model] = ...
%     LBL_Module_LinearDynamics_bprop(zHistory, dL_dz, thetas, model)
% Inputs:
%   zHistory: matrix of size <dim_z> x <n> x <n_samples> of latent history
%   dL_dz:    matrix of size <dim_z> x <n_samples> of gradient
%             on the latent linear predictions
%   thetas:   matrix of topic mixture weights of size <dim_k> x <n_samples>
%   model:    struct containing the model
% Outputs:
%   dL_dzHistory: matrix of size <dim_z> x <n> x <n_samples> of gradients
%                 on the latent variables' history (n-gram)
%   model:        struct containing the model

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
  LBL_Module_LinearDynamics_bprop(zHistory, dL_dz, thetas, model)

% Number of predictions (samples) which were made
n_samples = size(dL_dz, 2);

% Use a stacked history and single C matrix

% Derivatives w.r.t. stacked histories of latent variables
if (model.n_topics == 0)
  % Single topic (single dynamics)
  % C' has size (<dim_z> * <n>) x <dim_z>
  % dL_dz has size <dim_z> x <n_samples>
  % dL_dzHistory has size (<dim_z> * <n>) x <n_samples>
  dL_dzHistory = model.C' * dL_dz;
else
  % Multiple topics/dynamics with mixture weights
  coeffs = repmat(thetas(1, :), model.numel_zh, 1);
  dL_dzHistory = coeffs .* (model.C{1}' * dL_dz);
  for k = 2:model.n_topics
    coeffs = repmat(thetas(k, :), model.numel_zh, 1);
    dL_dzHistory = dL_dzHistory + coeffs .* (model.C{k}' * dL_dz);
  end
end
% Destack the history vectors into a sequence of <n> vectors
dL_dzHistory = reshape(dL_dzHistory, [model.dim_z, model.n, n_samples]);

if (nargout > 1)

  % Derivatives w.r.t. linear transformation matric C:
  % we need to add them for all samples
  if (model.n_topics == 0)
    % Single topic (single dynamics)
    % dL_dz has size <dim_z> x <n_samples>
    % zHistory' has size <n_samples> x (<dim_z> * <n>)
    % dL_dC has size <dim_z> x (<dim_z> * <n>)
    model.dL_dC = dL_dz * reshape(zHistory, model.numel_zh, n_samples)';
  else
    % Multiple topics/dynamics with mixture weights
    zHistory_reshaped = reshape(zHistory, model.numel_zh, n_samples)';
    for k = 1:model.n_topics
      coeffs = repmat(thetas(k, :), model.dim_zw, 1);
      model.dL_dC{k} = (coeffs .* dL_dz) * zHistory_reshaped;
    end
  end
  
  % Derivatives w.r.t. bias
  if (model.n_topics == 0)
    % Single topic (single dynamics)
    model.dL_dbC = sum(dL_dz, 2);
  else
    % Multiple topics/dynamics with mixture weights
    for k = 1:model.n_topics
      model.dL_dbC{k} = dL_dz * thetas(k, :)';
    end
  end
end
