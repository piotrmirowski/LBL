% LBL_Meter_Update  Update the meter with a new step's values
%
% Syntax:
%   meter = ...
%     LBL_Meter_Update(meter, L_e, eR_e, ppx_e, [L_m, eR_m, ppx_m, dL_sum])
% Inputs:
%   meter:  struct containing the evaluation meter
%   L_e:    vector of length <T> of prior word likelihood, before M-step
%   eR_e:   vector of length <T> of prior observation energy, before M-step
%   ppx_e:  vector of length <T> of prior perplexity, before M-step
%   L_m:    vector of length <T> of posterior word likelihood, after M-step
%   eR_m:   vector of length <T> of posterior observ. energy, after M-step
%   ppx_m:  vector of length <T> of posterior perplexity, after M-step
%   dL_sum: vector of length <T> of sums of likelihood increases in M-step
% Outputs:
%   meter: struct containing the evaluation meter

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

function meter = ...
  LBL_Meter_Update(meter, L_e, eR_e, ppx_e, L_m, eR_m, ppx_m, dL_sum)

if (nargin <= 1)
  if ((nargin == 1) && ischar(meter))
    % Initialize the meter without name
    meter = ...
      struct('name', meter, ...
      'n_steps', 0, 'L_e', [], 'eR_e', [], 'ppx_e', [], ...
      'L_m', [], 'eR_m', [], 'ppx_m', []);  
  else
    % Initialize the meter without name
    meter = ...
      struct('n_steps', 0, 'L_e', [], 'eR_e', [], 'ppx_e', [], ...
      'L_m', [], 'eR_m', [], 'ppx_m', []);
  end
  return;
end

n_steps = meter.n_steps + 1;
meter.n_steps = n_steps;
meter.L_e(n_steps) = mean(L_e);
meter.eR_e(n_steps) = mean(eR_e);
meter.ppx_e(n_steps) = ppx_e;
if (nargin > 4)
  meter.L_m(n_steps) = mean(L_m);
  meter.eR_m(n_steps) = mean(eR_m);
  meter.ppx_m(n_steps) = ppx_m;
end
if (nargin > 7)
  meter.dL_sum(n_steps) = mean(dL_sum);
end
