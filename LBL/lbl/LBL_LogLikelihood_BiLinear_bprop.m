% Compute the derivatives of the log-likelihood of the target word
% given the predicted word based on the word's history

% Syntax:
%   [dL_dzOutput, dL_dzBar, dL_dbw] = ...
%     LBL_LogLikelihood_BiLinear_bprop(zVocabulary, zBar, P, w)
% Inputs:
%   zVocabulary: matrix of size <M> x <V> of the latent representations of
%                all <V> words in the vocabulary
%   zBar:        matrix of size <M> x <T> of the predicted latent
%                representations of <T> words
%   P:           matrix of size <V> x <T> of predicted word probabilities
%   w:           vector of length <T> of target words
% Outputs:
%   dL_dzOutput: matrix of size <M> x <T> of gradients onto the target's
%                representation
%   dL_dzBar:    matrix of size <M> x <T> of gradients onto 
%                the prediction's (z_bar) representation
%   dL_dbw:      vector of size <V> x 1 of gradients onto the word biases

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

function [dL_dzOutput, dL_dzBar, dL_dbw] = ...
  LBL_LogLikelihood_BiLinear_bprop(zVocabulary, zBar, P, w)

n_samples = size(P, 2);
dim_w = size(P, 1);

% Derivative w.r.t. embedding <z> of word <w>
% indW = sub2ind([dim_w n_samples], w, 1:n_samples);
% dL_dzTarget = zBar * (1 - P(indW))';
% dL_dzTarget = zBar .* repmat(1 - P(indW), size(zBar, 1), 1);
dL_dzOutput = -zBar * P';
% Accumulate additional terms for gradients w.r.t. target word represent.
for k = 1:n_samples
  w_k = w(k);
  dL_dzOutput(:, w_k) = dL_dzOutput(:, w_k) + zBar(:, k);
end


% Derivative w.r.t. embedding <z> of word <w>
% dL_dbw = zeros(dim_w, 1);
dL_dbw = -sum(P, 2);
% Accumulate additional terms for gradients w.r.t. target word biases
for k = 1:n_samples
  w_k = w(k);
  dL_dbw(w_k) = dL_dbw(w_k) + 1;
end


% Expected embedding under predictive distribution <p> at each time point
% This represents T*(M*V) (<dim_w> x <dim_z> x <n_samples>) ops.
zMu = zVocabulary * P;

% Derivative w.r.t. embedding <zBar> (prediction)
dL_dzBar = zVocabulary(:, w) - zMu;


