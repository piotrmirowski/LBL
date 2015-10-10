% LBL_Dataset_AddTopics  Add topics to a dataset
%
% Syntax:
%   dataset = LBL_Dataset_AddTopics(dataset, theta, use_line)
% Inputs:
%   dataset:  struct containing the dataset with n-best hypotheses
%   theta:    matrix of size <n_topics> x <n_references> of LDA simplexes
%   use_line: boolean scalar: should one consider line-based topics instead
%             of document-based topics?
% Outputs:
%   dataset: struct containing the dataset with n-best hypotheses, with a 
%            topic assignement for each hypothesis

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

function dataset = LBL_Dataset_AddTopics(dataset, theta, use_line)

if (use_line)
  map = dataset.lines;
else
  map = dataset.documents;
end
t_min = min(map);
t_max = max(map);
n_words = length(dataset.wTargets);

% Check that there is a one-to-one correspondence in topics
n_topics = size(theta, 1);
n_samples = size(theta, 2);
if ((t_max - t_min + 1) ~= n_samples)
  error('Wrong number of documents/lines: %d', n_samples);
end
map = map - t_min + 1;

% Copy the topics to individual words
dataset.topics = theta(:, map)';
