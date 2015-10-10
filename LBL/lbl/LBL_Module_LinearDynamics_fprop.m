% LBL_Module_LinearDynamics_fprop  Predict latent vars with linear dynamics
%
% Syntax:
%   zLinear = LBL_Module_LinearDynamics_fprop(zHistory, thetas, model)
% Inputs:
%   zHistory: matrix of size <dim_z> x <n> x <n_samples> of latent history
%   thetas:   matrix of topic mixture weights of size <dim_k> x <n_samples>
%   model:    struct containing the model
% Outputs:
%   zLinear:  matrix of size <dim_z> x <n_samples> of latent predictions

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

function zLinear = LBL_Module_LinearDynamics_fprop(zHistory, thetas, model)

% Number of predictions (samples) which were made
n_samples = size(zHistory, 3);

% Use a stacked history and single C matrix:
% this enables multiple (<n_samples>) predictions at a time
if (model.n_topics == 0)
  % Single topic (single dynamics)
  zLinear = ...
    model.C * reshape(zHistory, model.numel_zh, n_samples) + ...
    repmat(model.Cbias, 1, n_samples);
else
  % Multiple topics/dynamics with mixture weights
  coeffs = repmat(thetas(1, :), model.dim_zw, 1);
  zHistory_reshape = reshape(zHistory, model.numel_zh, n_samples);
  zLinear = coeffs .* ...
    (model.C{1} * zHistory_reshape + repmat(model.Cbias{1}, 1, n_samples));
  for k = 2:model.n_topics
    coeffs = repmat(thetas(k, :), model.dim_zw, 1);
    zLinear = zLinear + coeffs .* ...
      (model.C{k} * zHistory_reshape + ...
      repmat(model.Cbias{k}, 1, n_samples));
  end
end
