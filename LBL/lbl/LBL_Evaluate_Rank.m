% LBL_Evaluate_Rank  Evaluate ranks of correct word prediction, print top 5
%
% Syntax:
%   ranks = LBL_Evaluate_Rank(dataset, model, params, fid)
% Inputs:
%   dataset: struct containing the dataset used for evaluation
%   model:   struct containing the model being evaluated
%   params:  struct containing the parameters
%   fid:     scalar containing the file identifier for output (1 = screen)
% Outputs:
%   ranks:   vector of length <T> of ranks of the correct predictions

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

function ranks = LBL_Evaluate_Rank(dataset, model, params, fid)

% Prepare the order of the training dataset
dataset = LBL_Dataset_Init(dataset, params, 0);

n_words = length(dataset.vocabulary);
n_samples = length(dataset.wTargets);
ranks = zeros(1, n_samples);

% Evaluate the rank statistics per block
for k = 1:dataset.n_batches

  % Get batch of data
  [wTarget_k, wHistory_k, xHistory_k, thetas_k, indK] = ...
    LBL_Dataset_fprop(dataset, params, k);
  % Make the prediction from the (n-1)-gram histories
  Pk = LBL_Model_fprop(wTarget_k, wHistory_k, xHistory_k, ...
    model.R, thetas_k, model, params);
  % Ranks of the correct predictions
  for i = 1:length(indK)
    [p_ki, ind_ki] = sort(Pk(:, i), 'descend');
    ranks(indK(i)) = find(ind_ki == wTarget_k(i));
    for j = 1:model.n
      fprintf(fid, '%s ', model.vocabulary{wHistory_k(j, i)});
    end
    fprintf(fid, '[%s]', model.vocabulary{wTarget_k(i)});
    for j = 1:5
      fprintf(fid, ' %s', model.vocabulary{ind_ki(j)});
    end
    fprintf(fid, '\n');
  end

  % Trace
  if (mod(k, 10) == 0)
    fprintf(1, 'o');
  else
    fprintf(1, '.');
  end
end

% Count statistics on the prediction ranks
cntBins = unique([1 2 3 4 5 10:10:50 100 500:500:n_words n_words]);
n_bins = length(cntBins);
cnts = zeros(1, n_bins);
cnts(1) = sum(ranks <= cntBins(1));
labels = cell(1, n_bins);
labels{1} = sprintf('#=%d: %.1f%%', cntBins(1), 100*cnts(1)/n_samples);
for k = 2:n_bins
  cnts(k) = sum((ranks > cntBins(k-1)) & (ranks <= cntBins(k)));
  if (cntBins(k-1) + 1 == cntBins(k))
    labels{k} = sprintf('#=%d: %.1f%%', cntBins(k), 100*cnts(k)/n_samples);
  else
    labels{k} = sprintf('%d<#<=%d %.1f%%', ...
      cntBins(k-1), cntBins(k), 100*cnts(k)/n_samples);
  end
end
pie(cnts, labels);
