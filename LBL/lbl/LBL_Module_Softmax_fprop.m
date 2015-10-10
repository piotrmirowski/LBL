% LBL_Module_Softmax_fprop  Softmax
%
% Syntax:
%   p = LBL_Module_Softmax_fprop(e, [sign_e])
% Inputs:
%   e:      matrix of size <V> x <T> of word energies at each time point
%   sign_e: sign (scalar), by default equal to 1 (positive)
% Outputs:
%   p:      matrix of size <V> x <T> of word probability at each time point

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

function p = LBL_Module_Softmax_fprop(e, sign_e)

% Compute the numerator
if ((nargin == 1) || (sign_e > 0))
  p = exp(e);
else
  p = exp(-e);
end
  
% Compute the denominator
den = sum(p);
n_samples = size(p, 2);
if (n_samples > 1)

  % Multi-sample softmax
  if 1

    % Loop over samples (faster?)
    for k = 1:n_samples
      p(:, k) = p(:, k) / den(k);
    end
  else

    % Loop over words (slower?)
    dim_w = size(p, 1);
    for k = 1:dim_w
      p(k, :) = p(k, :) ./ den;
    end
  end
else
  
  % Single-sample softmax
  p = p / den;
end
