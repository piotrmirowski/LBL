% LBL_Model_SaveBest  Keep best model (w.r.t. a given measure) in history
%
% Syntax:
%   model = LBL_Model_SaveBest(model, measure, epoch)
% Inputs:
%   model:       struct containing the model
%   measure:     perplexity measure achieved at the latest <epoch>
%   epoch:       latest learning iteration/epoch
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
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu
%
% (c) 2010, AT&T Labs
%     180 Park Avenue, Florham Park, NJ 07932, USA.

function model = LBL_Model_SaveBest(model, measure, epoch)

if (model.measure > measure)

  % Save best parameters
  model.R_best = model.R;
  model.bw_best = model.bw;
  model.C_best = model.C;
  model.Cbias_best = model.Cbias;
  model.epoch_best = epoch;
  model.measure = measure;
  if (model.dim_h > 0)
    model.A_best = model.A;
    model.Abias_best = model.Abias;
    model.B_best = model.B;
    model.Bbias_best = model.Bbias;
  end
  if (model.dim_x > 0)
    model.F_best = model.F;
    model.Fbias_best = model.Fbias;
  end
end
