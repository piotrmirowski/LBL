% LBL_Estep  (Stochastically) learn the model parameters on a mini-batch
%
% Syntax:
%   [L, eR, ppx, zVocabulary, P] = ...
%     LBL_Estep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
%     model, params)
% Inputs:
%   wTarget:     vector of size 1 x <n_samples> of target word indexes
%   wHistory:    matrix of size <n> x <n_samples> of n-gram word indexes
%   xHistory:    matrix of size <dim_x> x <n> x <n_samples> of features
%   zVocabulary: matrix of size <dim_z> x <dim_w> of vocabulary embeddding
%   thetas:      matrix of topic mixtures of size <dim_k> x <n_samples>
%   model:       struct containing the model after 1 learning step
%   params:      struct containing the parameters
% Outputs:
%   L:           vector of length <n_samples> of prior log-likelihoods
%   eR:          vector of length <n_samples> of prior observation errors
%   ppx:         scalar containing the prior perplexity
%   zVocabulary: matrix of size <dim_z> x <dim_w> of vocabulary embeddding
%                after 1 M-step
%   P:           matrix of size <dim_w> x <n_samples> of prior
%                word probabilities

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

function [L, eR, ppx, zVocabulary, P] = ...
  LBL_Estep(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
  model, params)


% Forward propagation to get the prediction, log-likelihoods, etc...
[P, L, eR, zBar, zHistory, zHidden] = ...
  LBL_Model_fprop(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
  model, params);
ppx = LBL_LogLikelihood_Perplexity(L);

if (params.relax_z)

  % The code below was never used...
  
  n_samples = length(wTarget);
  
  % Copy of current latent variables
  zCopy = zVocabulary;

  % Relaxation until convergence
  cond = 1;
  n_e_steps = 1;
  while (cond)
  
    % Gradient to the latent variables over all the vocabulary
    [dL_dzVocabulary, wAll] = ...
      LBL_Model_bprop(P, zBar, zHistory, zVocabulary, zHidden, ...
      wTarget, wHistory, xHistory, thetas, model, params);

    % Gradient ascent
    zVocabulary(:, wAll) = ...
      zVocabulary(:, wAll) + eta_z * dL_dzVocabulary(:, wAll);

    % Forward propagation to get the prediction, log-likelihoods, etc...
    [P, Lnew, eRnew, zBar, zHistory, zHidden] = ...
      LBL_Model_fprop(wTarget, wHistory, xHistory, zVocabulary, thetas, ...
      model, params);
    ppx_new = LBL_LogLikelihood_Perplexity(Lnew);

    fprintf(1, 'E-step %3d: L=%10.4f (ppx=%10g), eR=%10.4f\n', ...
      n_e_steps, sum(Lnew), ppx_new, sum(eR));
    
    % Evaluate the convergence
    if (sum(Lnew) < sum(L))

      % Retrieve previous state
      cond = 0;
      zVocabulary(:, wAll) = zCopy(:, wAll);
    else

      % Validate
      n_e_steps = n_e_steps + 1;
      zCopy(:, wAll) = zVocabulary(:, wAll);
      L = Lnew;
      eR = eRnew;
      ppx = ppx_new;
    end
  end
end
