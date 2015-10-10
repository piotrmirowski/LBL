% LBL_Module_Encoder_bprop  Back-propagation to the word encoding
%
% Syntax:
%   model = LBL_Module_Encoder_bprop(w, dL_dz, model)
% Inputs:
%   w:        vector of length <n_samples> of word indexes
%   dL_dz:    matrix of size <dim_z> x <n_samples> of gradient
%             on the word representations
%   model:    struct containing the model
% Outputs:
%   model:    struct containing the model with updated gradients

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

function model = LBL_Module_Encoder_bprop(w, dL_dz, model)

% Apply the gradients <dL_dz> to the columns of <R> selected by the word
% indexes that generated the <z>
model.dL_dR = zeros(model.dim_zw, model.dim_w);
model.dL_dR(:, w) = dL_dz;
