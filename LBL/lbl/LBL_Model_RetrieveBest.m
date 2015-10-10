% LBL_Model_RetrieveBest  Replace parameters by best historical parameters
%
% Syntax:
%   model = LBL_Model_RetrieveBest(model)
% Inputs:
%   model:       struct containing the model
% Outputs:
%   model:       struct containing the model

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

function model = LBL_Model_RetrieveBest(model)

% Retrieve best parameters
model.R = model.R_best;
model.bw = model.bw_best;
model.C = model.C_best;
model.Cbias = model.Cbias_best;
if (model.dim_h > 0)
  model.A = model.A_best;
  model.Abias = model.Abias_best;
  model.B = model.B_best;
  model.Bbias = model.Bbias_best;
end
if (model.dim_x > 0)
  model.F = model.F_best;
  model.Fbias = model.Fbias_best;
end
