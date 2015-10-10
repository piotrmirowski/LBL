% LBL_Evaluate_NBest  Evaluate a language model on a dataset of n-best hypo
%
% Syntax:
%   [meter, dataset] = ...
%     LBL_Evaluate_NBest(model, dataset, params, ...
%       [with_ref, amCost, coeff_lm])
% Inputs:
%   model:    struct containing the model being evaluated
%   dataset:  struct containing the dataset used for evaluation
%   params:   struct containing the parameters
%   with_ref: boolean scalar: shall one include references among hypotheses
%   amCost:   vector of length <n_nGrams> of acoustic model costs
%             (this vector needs to be properly normalized at word level)
%   coeff_lm: scalar coefficient multiplicative of the language model score
% Outputs:
%   meter:   struct containing the updated evaluation meter
%   dataset: struct containing the dataset used for evaluation

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

function [meter, dataset] = ...
  LBL_Evaluate_NBest(model, dataset, params, force_ref, amCost, coeff_lm)


% Be sure that we have the best model
model = LBL_Model_RetrieveBest(model);

% By default, do not include reference in the choices
if (nargin < 4)
  force_ref = 0;
end


% Language model likelihood
if ~isfield(dataset, 'L')
  % Evaluate the entire dataset if the log-likelihood is not yet computed
  [meter, L] = LBL_Evaluate(model, dataset, params);
  
  % Store the log-likelihood
  dataset.L = L;
else
  meter = [];
  L = dataset.L;
end


% Combination with the acoustic model cost
if (nargin == 6)
  use_am = 1;
  L = L * coeff_lm - amCost;
  ext_am = '_am';
else
  use_am = 0;
  ext_am = '';
end





% Prepare the order of the evaluation dataset
dataset = LBL_Dataset_Init(dataset, params, 0);


% Open the evaluation files
file_ref = [params.filename ext_am '_nbest_ref.txt'];
if (force_ref)
  file_hypo = [params.filename ext_am '_nbest_hypo.with_ref.txt'];
else
  file_hypo = [params.filename ext_am '_nbest_hypo.txt'];
end
fid_ref = fopen(file_ref, 'w');
fid_hypo = fopen(file_hypo, 'w');
if (force_ref)
  file_debug = [params.filename ext_am '_nbest_hypo.debug.txt'];
  fid_debug = fopen(file_debug, 'w');
end

% Sum the log-likelihood per reference and per choice
vocabulary = dataset.vocabulary;
n_ref = length(unique(dataset.nBestRef));
for k = 1:n_ref
  ind_k = (dataset.nBestRef == k);
  choices_k = unique(dataset.nBestChoice(ind_k));
  meanL_k = zeros(1, max(choices_k));
  wTargets_k = dataset.wTargets(ind_k);
  nBestChoice_k = dataset.nBestChoice(ind_k);
  L_k = L(ind_k);

  % Loop over the choices: reference and hypotheses
  for c = choices_k(:)'
    ind_kc = find(nBestChoice_k == c);
    if (c == 0)
      % Handle the reference
      meanL_ref_k = SentenceLL(L_k(ind_kc), use_am);
    else
      % Handle each hypothesis
      meanL_k(c) = SentenceLL(L_k(ind_kc), use_am);
    end
  end

  % Choose the hypothesis with minimal mean negative log-likelihood
  [meanL_k_best, best_kc] = max(meanL_k);

  % Override the hypothesis with reference, when lower perplexity?
  try
    if (force_ref && (meanL_ref_k > meanL_k_best))
      best_kc = 0;
    end
  catch
    best_kc = [];
  end

  % Write the reference
  ind_k0 = find(nBestChoice_k == 0);
  strRef_k = '';
  for i = ind_k0(:)'
    strRef_k = sprintf('%s%s ', strRef_k, vocabulary{wTargets_k(i)});
  end
  fprintf(1, 'Reference (nLL=%g):\n%s\n', -meanL_ref_k, strRef_k);
  fprintf(fid_ref, '%s\n', strRef_k);

  % Write the best choices
  ind_kc_best = find(nBestChoice_k == best_kc);
  strBest_k = '';
  for i = ind_kc_best(:)'
    strBest_k = sprintf('%s%s ', strBest_k, vocabulary{wTargets_k(i)});
  end
  fprintf(1, 'Best hypothesis (nLL=%g):\n%s\n', -meanL_k_best, strBest_k);
  fprintf(fid_hypo, '%s\n', strBest_k);

  % Debug in the case the reference is not chosen
  if (force_ref && ~isempty(best_kc) && (best_kc > 0))

    % Write the reference, n-gram by n-gram
    ind_k0 = find(nBestChoice_k == 0);
    fprintf(fid_debug, 'Reference %d (nLL=%g):\n', k, -meanL_ref_k);
    for i = ind_k0(:)'
      fprintf(fid_debug, '%s: %.2f\n', vocabulary{wTargets_k(i)}, L_k(i));
    end

    % Write the best choices
    ind_kc_best = find(nBestChoice_k == best_kc);
    fprintf(fid_debug, 'Hypothesis for %d (nLL=%g):\n', k, -meanL_k_best);
    for i = ind_kc_best(:)'
      fprintf(fid_debug, '%s: %.2f\n', vocabulary{wTargets_k(i)}, L_k(i));
    end
    fprintf(fid_debug, '\n');
  end
end
fclose(fid_ref);
fclose(fid_hypo);


% -------------------------------------------------------------------------
function res_ll = SentenceLL(L, use_am)

ind = ~isnan(L);
res_ll = sum(L(ind));
if (use_am)
  res_ll = res_ll / sum(ind);
end
