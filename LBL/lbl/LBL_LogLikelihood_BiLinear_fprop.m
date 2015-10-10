% LBL_LogLikelihood_BiLinear_fprop  Likelihood in logbilinear energy model
%
% Syntax:
%   [L, P] = LBL_LogLikelihood_BiLinear_fprop(w, zVocabulary, zBar, model)
% Inputs:
%   w:           vector of length <T> of target words
%   zVocabulary: matrix of size <M> x <V> of the latent representations of
%                all <V> words in the vocabulary
%   zBar:        matrix of size <M> x <T> of the predicted latent
%                representations of <T> words
%   model:       struct containing the model
% Outputs:
%   L: vector of length <T> of log-likelihood values
%   P: matrix of size <V> x <T> of word probabilities over <V> words
%      for <T> samples

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

function [L, P] = ...
  LBL_LogLikelihood_BiLinear_fprop(w, zVocabulary, zBar, model)


% Number of predictions
n_samples = size(zBar, 2);

% Bilinear energy between all words' encodings and all predictions
% N.B.: we do not take the negative of the negative...
% This represents T*V*M (<dim_w> x <dim_z> x <n_samples>) ops.
E = (zVocabulary' * zBar) + repmat(model.bw, 1, n_samples);

% Normalize the energy using softmax to get word probabilities
P = LBL_Module_Softmax_fprop(E);

% Log-likelihood at the actually observed words <w>
n_samples = length(w);
L = zeros(1, n_samples);

if isfield(model, 'tag_unk')

  % Open-vocabulary evaluation
  for k = 1:n_samples
    if (w(k) ~= model.tag_unk)
      L(k) = log(P(w(k), k));
    else
      L(k) = nan;
    end
  end
else
  
  % Closed-vocabulary evaluation
  for k = 1:n_samples
    L(k) = log(P(w(k), k));
  end
end