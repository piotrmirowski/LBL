% LBL_Module_History_fprop  Concatenate the latent variables into history
%
% Syntax:
%   zHistory = LBL_Module_History_fprop(w, zVocabulary, model, [zx])
% Inputs:
%   w:           vector of length <N> x <T> of word indices in <T> <N>grams
%   zVocabulary: matrix of size <M> x <V> of the latent representations of
%                all <V> words in the vocabulary
%   model:       struct containing the model
%   zx:          (optional) matrix of size <F> x <TxN> of feature embedding
% Outputs:
%   zHistory: matrix of size <M'> x <N> x <T> of the latent representations
%             of <T> <N>-grams in the <M'>-dimensional latent space,
%             where <M'> = <M> + <F>

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

function zHistory = LBL_Module_History_fprop(w, zVocabulary, model, zx)

n_samples = size(w, 2);

% Get the history of word embeddings as a single matrix
w = reshape(w, 1, model.n * n_samples);
zw = zVocabulary(:, w);

if (nargin < 4)

  % Transform the history of word embeddings into a 3D array
  zHistory = reshape(zw, model.dim_zw, model.n, n_samples);
else
  
  % Add the history of feature embeddings
  zHistory = ...
    reshape([zw; zx], (model.dim_zw + model.dim_zx), model.n, n_samples);
end
