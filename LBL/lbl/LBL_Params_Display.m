% LBL_Params_Display  Print the hyper-parameters to a file or to the screen
%
% Syntax:
%   LBL_Params_Display(params, fid)
% Inputs:
%   params: struct containing the parameters
%   fid:    file identifier (1=stdout by default)

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

function LBL_Params_Display(params, fid)

if (nargin < 2)
  fid = 1;
end
fprintf(fid, 'params,dim_w,%d\nparams,dim_x,%d\nparams,n,%d\n', ...
  params.dim_w, params.dim_x, params.n);
fprintf(fid, 'params,dim_zw,%d\nparams,dim_zx,%d\nparams,dim_z,%d\n', ...
  params.dim_zw, params.dim_zx, params.dim_z);
fprintf(fid, 'params,dim_h,%d\n', params.dim_h);
fprintf(fid, 'params,n_epochs,%d\nparams,len_batch,%d\n', ...
  params.n_epochs, params.len_batch);
fprintf(fid, 'params,init_r,%s\n', params.init_r);
fprintf(fid, 'params,eta_w,%g\nparams,eta_r,%g\nparams,eta_c,%g\n', ...
  params.eta_w, params.eta_r, params.eta_c);
fprintf(fid, 'params,eta_a,%g\nparams,eta_b,%g\n', ...
  params.eta_a, params.eta_b);
fprintf(fid, 'params,eta_f,%g\n', params.eta_f);
try
  fprintf(fid, 'params,approx_ll,%d\n', params.approx_ll);
  fprintf(fid, 'params.n_neighb_approx_ll,%d\n', params.n_neighb_approx_ll);
  fprintf(fid, 'params.n_approx_ll,%d\n', params.n_approx_ll);
end
fprintf(fid, 'params,eta_w0,%g\nparams,eta_r0,%g\nparams,eta_c0,%g\n', ...
  params.eta_w0, params.eta_r0, params.eta_c0);
fprintf(fid, 'params,eta_a0,%g\nparams,eta_b0,%g\n', ...
  params.eta_a0, params.eta_b0);
fprintf(fid, 'params,eta_f0,%g\n', params.eta_f0);
fprintf(fid, 'params,eta_anneal,%g\n', params.eta_anneal);
fprintf(fid, 'params,lambda_r,%g\nparams,lambda_c,%g\n', ...
  params.lambda_r, params.lambda_c);
fprintf(fid, 'params,lambda_a,%g\nparams,lambda_b,%g\n', ...
  params.lambda_a, params.lambda_b);
fprintf(fid, 'params,lambda_f,%g\n', params.lambda_f);
fprintf(fid, 'params,gamma,%g\n', params.gamma);
fprintf(fid, 'params,momentum,%g\n', params.momentum);
fprintf(fid, 'params,relax_z,%d\nparams,eta_z,%g\n', ...
  params.relax_z, params.eta_z);
fprintf(fid, 'params,n_topics,%d\nparams,topic_model,%s\n', ...
  params.n_topics, params.topic_model);
