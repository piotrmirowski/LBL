% LBL_Learn  (Stochastically) learn the model parameters on a mini-batch
%
% Syntax:
%   [model, meter] = ...
%     LBL_Learn(wTarget, wHistory, xHistory, thetas, model, params, meter)
% Inputs:
%   wTarget:  vector of size 1 x <n_samples> of target word indexes
%   wHistory: matrix of size <n> x <n_samples> of n-gram word indexes
%   xHistory: matrix of size <dim_x> x <n> x <n_samples> of features
%   thetas:   matrix of topic mixture weights of size <dim_k> x <n_samples>
%   model:    struct containing the model after 1 learning step
%   params:   struct containing the parameters
%   meter:    struct containint the learning meter
% Outputs:
%   model:    struct containing the model after 1 learning step
%   meter:    struct containint the learning meter

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

function [model, meter] = ...
  LBL_Learn(wTarget, wHistory, xHistory, thetas, model, params, meter)

% Switch to learning mode
params.eval_mode = 0;

% Latent variables
zVocabulary = model.R;

if (params.relax_z)
  % E-step to produce the latent representation by relaxation
  [L_e, eR_e, ppx_e, zVocabulary] = ...
    LBL_Estep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
    model, params);
end

if (params.relax_z)

  % M-step to optimize the parameters
  [L_m, eR_m, ppx_m, model, dL_sum] = ...
    LBL_Mstep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
    model, params);

  % Store the results in the meter
  meter = ...
    LBL_Meter_Update(meter, L_e, eR_e, ppx_e, L_m, eR_m, ppx_m, dL_sum);
else
  
  % M-step to optimize the parameters and produce latent representation R
  [L_e, eR_e, ppx_e, model, dL_sum, L_m, eR_m, ppx_m] = ...
    LBL_Mstep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
    model, params);

  % Store the results in the meter
  if (params.eval_ppx_after)
    meter = ...
      LBL_Meter_Update(meter, L_e, eR_e, ppx_e, L_m, eR_m, ppx_m, dL_sum);
  else
    meter = LBL_Meter_Update(meter, L_e, eR_e, ppx_e);
  end
end

