% LBL_Mstep  (Stochastically) learn the model parameters on a mini-batch
%
% Syntax:
%   [Linit, eRinit, ppx_init, model, dL_sum, L, eR, ppx] = ...
%     LBL_Mstep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
%               model, params)
% Inputs:
%   wTarget:     vector of size 1 x <n_samples> of target word indexes
%   wHistory:    matrix of size <n> x <n_samples> of n-gram word indexes
%   xHistory:    matrix of size <dim_x> x <n> x <n_samples> of features
%   zVocabulary: matrix of size <dim_z> x <dim_w> of vocabulary embeddding
%   thetas:      matrix of topic mixtures of size <dim_k> x <n_samples>
%   model:       struct containing the model after 1 learning step
%   params:      struct containing the parameters
% Outputs:
%   Linit:       vector of length <n_samples> of prior log-likelihoods
%   eRinit:      vector of length <n_samples> of prior observation errors
%   ppx_init:    scalar containing the prior perplexity
%   model:       struct containing the model after 1 learning step
%   dL_sum:      sum of the deltas of log-likelihood over the mini-batch
%   L:           vector of length <n_samples> of posterior log-likelihoods
%   eR:          vector of length <n_samples> of posterior observation err.
%   ppx:         scalar containing the posterior perplexity

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

function [Linit, eRinit, ppx_init, model, dL_sum, L, eR, ppx] = ...
  LBL_Mstep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
  model, params)

% To monitor time
tic;


% -------------------------------------------
% Forward pass #1: prediction, log-likelihood
% -------------------------------------------

% Forward propagation to get the prediction, log-likelihoods, etc...
[P, Linit, eRinit, zBar, zHistory, zHidden] = ...
  LBL_Model_fprop(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
  model, params);
ppx_init = LBL_LogLikelihood_Perplexity(Linit);

% Trace
if (params.approx_ll), fprintf(1, '~'); end
% fprintf(1, 'M-step res: nLL=%10.2f ppx=%6.1f eR=%6.1f', ...
%   -sum(Linit), ppx_init, sum(eRinit));
fprintf(1, 'M-step res: nLL=%10.2f ppx=%6.1f (%.2fs)', ...
  -sum(Linit), ppx_init, toc);


% ------------------------------------
% Backward pass: Compute the gradients
% ------------------------------------

[dL_dzVocabulary, wAll, model] = ...
  LBL_Model_bprop(P, zBar, zHistory, zVocabulary, zHidden, ...
  wTarget, wHistory, xHistory, thetas, model, params);


% -----------------------
% Gradient-based learning
% -----------------------

model = ApplyGradients(model, params, wAll);


% ----------------------------------------------------------
% Forward pass #2: evaluation after learning (not necessary)
% ----------------------------------------------------------

if (params.eval_ppx_after)

  % Re-evaluate the log-likelihood and reconstruction error
  [P, L, eR] = ...
    LBL_Model_fprop(wTarget, wHistory, xHistory, ...
    zVocabulary, thetas, model, params);
  ppx = LBL_LogLikelihood_Perplexity(L);

  % Improvement in the log-likelihood
  dL_sum = sum(L) - sum(Linit);

  % Trace
  fprintf(1, ' -> nLL=%10.2f ppx=%6.1f eR=%6.1f ', ...
    -sum(L), ppx, sum(eR));
  if (dL_sum > 0), fprintf(1, ' \\'); else fprintf(1, ' //'); end
else

  % Do not re-evaluate the change
  ppx = 0;
  dL_sum = 0;
  L = [];
  eR = [];
end
fprintf(1, ' (%.2fs)\n', toc);


% -------------------------------------------------------------------------
function model = ApplyGradients(model, params, wAll)


% Momentum term on the dynamical parameters
momentum = params.momentum;
if (momentum > 0)
  if isfield(model, 'dL_dbC_prev')

    % 1) Add momentum term
    if (params.use_lin)
      % 1a) Linear dynamics
      if (model.n_topics == 0)
        % Single dynamics
        model.dL_dC = ...
          model.dL_dC * (1 - momentum) + model.dL_dC_prev * momentum;
        model.dL_dbC = ...
          model.dL_dbC * (1 - momentum) + model.dL_dbC_prev * momentum;
      else
        % Multiple topics/dynamics
        for k = 1:model.n_topics
          model.dL_dC{k} = ...
            model.dL_dC{k} * (1 - momentum) + model.dL_dC_prev{k} * momentum;
          model.dL_dbC{k} = ...
            model.dL_dbC{k} * (1 - momentum) + ...
            model.dL_dbC_prev{k} * momentum;
        end
      end
    end
    if (model.dim_h > 0)
      % 1b) Nonlinear dynamics
      if (model.n_topics == 0)
        % Single dynamics
        model.dL_dA = ...
          model.dL_dA * (1 - momentum) + model.dL_dA_prev * momentum;
        model.dL_dbA = ...
          model.dL_dbA * (1 - momentum) + model.dL_dbA_prev * momentum;
        if ~isempty(model.B)
          model.dL_dB = ...
            model.dL_dB * (1 - momentum) + model.dL_dB_prev * momentum;
          model.dL_dbB = ...
            model.dL_dbB * (1 - momentum) + model.dL_dbB_prev * momentum;
        end
      else
        % Multiple topics/dynamics
        for k = 1:model.n_topics
          model.dL_dA{k} = ...
            model.dL_dA{k} * (1 - momentum) + ...
            model.dL_dA_prev{k} * momentum;
          model.dL_dbA{k} = ...
            model.dL_dbA{k} * (1 - momentum) + ...
            model.dL_dbA_prev{k} * momentum;
          if ~isempty(model.B)
            model.dL_dB{k} = ...
              model.dL_dB{k} * (1 - momentum) + ...
              model.dL_dB_prev{k} * momentum;
            model.dL_dbB{k} = ...
              model.dL_dbB{k} * (1 - momentum) + ...
              model.dL_dbB_prev{k} * momentum;
          end
        end
      end
    end
    if ((model.dim_x > 0) && (model.dim_zx ~= model.dim_x))
      % 1c) Features
      model.dL_dF = ...
        model.dL_dF * (1 - momentum) + model.dL_dF_prev * momentum;
      model.dL_dbF = ...
        model.dL_dbF * (1 - momentum) + model.dL_dbF_prev * momentum;
    end
  end

  % 2) Store the gradients for next iteration
  if (params.use_lin)
    % 2a) Linear dynamics
    model.dL_dC_prev = model.dL_dC;
    model.dL_dbC_prev = model.dL_dbC;
  end
  if (model.dim_h > 0)
    % 2b) Nonlinear dynamics
    model.dL_dA_prev = model.dL_dA;
    model.dL_dbA_prev = model.dL_dbA;
    if ~isempty(model.B)
      model.dL_dB_prev = model.dL_dB;
      model.dL_dbB_prev = model.dL_dbB;
    end
  end
  if ((model.dim_x > 0) && (model.dim_zx ~= model.dim_x))
    % 2c) Features
    model.dL_dF_prev = model.dL_dF;
    model.dL_dbF_prev = model.dL_dbF;
  end
end
  
% Gradient descent
% The regularization is applied only to the hidden representation of 
% selected words, and not to the biases
if (params.use_lin)
  % a) Linear dynamics
  if (model.n_topics == 0)
    % Single dynamics
    model.C = (1 - params.lambda_c) * model.C + params.eta_c * model.dL_dC;
    model.Cbias = model.Cbias + params.eta_c * model.dL_dbC;
  else
    % Multiple topics/dynamics
    for k = 1:model.n_topics
      model.C{k} = (1 - params.lambda_c) * model.C{k} + ...
        params.eta_c * model.dL_dC{k};
      model.Cbias{k} = model.Cbias{k} + params.eta_c * model.dL_dbC{k};
    end
  end
end
if (model.dim_h > 0)
  % b) Nonlinear dynamics
  if (model.n_topics == 0)
    % Single dynamics
    model.A = (1 - params.lambda_a) * model.A + params.eta_a * model.dL_dA;
    model.Abias = model.Abias + params.eta_a * model.dL_dbA;
    if ~isempty(model.B)
      model.B = (1 - params.lambda_b) * model.B + params.eta_b * model.dL_dB;
      model.Bbias = model.Bbias + params.eta_b * model.dL_dbB;
    end
  else
    % Multiple topics/dynamics
    for k = 1:model.n_topics
      model.A{k} = ...
        (1 - params.lambda_a) * model.A{k} + params.eta_a * model.dL_dA{k};
      model.Abias{k} = ...
        model.Abias{k} + params.eta_a * model.dL_dbA{k};
      if ~isempty(model.B)
        model.B{k} = ...
          (1 - params.lambda_b) * model.B{k} + params.eta_b * model.dL_dB{k};
        model.Bbias{k} = ...
          model.Bbias{k} + params.eta_b * model.dL_dbB{k};
      end
    end
  end
end
if ((model.dim_x > 0) && (model.dim_zx ~= model.dim_x))
  model.F = (1 - params.lambda_f) * model.F + params.eta_f * model.dL_dF;
  model.Fbias = model.Fbias + params.eta_f * model.dL_dbF;
end
% c) Word embedding
model.R(:, wAll) = ...
  (1 - params.lambda_r) * model.R(:, wAll) + ...
  params.eta_r * model.dL_dR(:, wAll);
model.bw = model.bw + params.eta_r * model.dL_dbw;
