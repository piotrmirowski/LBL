% LBL_Meter_Display  Print the meter history into a file or screen
%
% Syntax:
%   LBL_Meter_Display(meter, [fid])
% Inputs:
%   meter:   struct containing the evaluation meter
%   fid:     file identifier for the output (by default 1 for stdout)

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

function LBL_Meter_Display(meter, fid)

if (nargin < 2)
  fid = 1;
end

for k = 1:meter.n_steps
  fprintf(fid, '%s,L,%d,%g\n', meter.name, k, meter.L_e(k));
end
for k = 1:meter.n_steps
  fprintf(fid, '%s,ppx,%d,%g\n', meter.name, k, meter.ppx_e(k));
end
if any(meter.eR_e)
  for k = 1:meter.n_steps
    fprintf(fid, '%s,eR,%d,%g\n', meter.name, k, meter.eR_e(k));
  end
end
