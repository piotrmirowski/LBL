% LBL_Energy_BiLinear_fprop  Bilinear energy
%
% Syntax:
%   e = LBL_Energy_BiLinear_fprop(zw, zBar, w, model)
% Inputs:
%   zw:    vector of size <M> x 1 of the latent representation of word <w>
%   zBar:  matrix of size <M> x <T> of the predicted latent representations
%   w:     target word index
%   model: struct containing the model
% Outputs:
%   e: vector of size 1 x <T> of bilinear energies between <T> samples and
%      the representation <z> of word <w>

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

function e = LBL_Energy_BiLinear_fprop(zw, zBar, w, model)

% Word bias
bw = model.bw;

% Bilinear energy: we assume that we have a sequence of targets <z>
% and associated predictions <zBar>
e = -(zw' * zBar) - bw(w);
