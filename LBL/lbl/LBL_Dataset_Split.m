% LBL_Dataset_Split  Split a dataset into learning and cross-validation set
%
% Syntax:
%   [datasetLearn, datasetXval] = LBL_Dataset_Split(datasetTrain, x)
% Inputs:
%   datasetTrain: struct containing the dataset used for training
%   x:            scalar between 0 and 1 indicating the fraction of data
%                 used for learning
% Outputs:
%   datasetLearn: struct containing the dataset used for learning
%   datasetTest:  struct containing the dataset used for cross-validation

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

function [datasetLearn, datasetXval] = LBL_Dataset_Split(datasetTrain, x)

if ((x < 0) || (x > 1))
  error('<x> needs to be between 0 and 1');
end

n_samples = length(datasetTrain.wTargets);
ind = randperm(n_samples);
n_learn = round(n_samples * x);
indLearn = ind(1:n_learn);
indXval = ind((n_learn+1):end);

datasetLearn = datasetTrain;
datasetLearn.wTargets = datasetLearn.wTargets(indLearn);
datasetLearn.wHistories = datasetLearn.wHistories(indLearn, :);
datasetXval = datasetTrain;
datasetXval.wTargets = datasetXval.wTargets(indXval);
datasetXval.wHistories = datasetXval.wHistories(indXval, :);
if isfield(datasetLearn, 'tagHistories')
  datasetLearn.tagHistories = datasetLearn.tagHistories(indLearn, :);
  datasetXval.tagHistories = datasetXval.tagHistories(indXval, :);
end
if isfield(datasetLearn, 'posTagHistories')
  datasetLearn.posTagHistories = datasetLearn.posTagHistories(indLearn, :);
  datasetXval.posTagHistories = datasetXval.posTagHistories(indXval, :);
end
if isfield(datasetLearn, 'xTagHistories')
  datasetLearn.xTagHistories = datasetLearn.xTagHistories(indLearn, :);
  datasetXval.xTagHistories = datasetXval.xTagHistories(indXval, :);
end
if isfield(datasetLearn, 'documents')
  datasetLearn.documents = datasetLearn.documents(indLearn);
  datasetXval.documents = datasetXval.documents(indXval);
end
if isfield(datasetLearn, 'sentences')
  datasetLearn.sentences = datasetLearn.sentences(indLearn);
  datasetXval.sentences = datasetXval.sentences(indXval);
end
if isfield(datasetLearn, 'lines')
  datasetLearn.lines = datasetLearn.lines(indLearn);
  datasetXval.lines = datasetXval.lines(indXval);
end
if isfield(datasetLearn, 'topics')
  datasetLearn.topics = datasetLearn.topics(indLearn, :);
  datasetXval.topics = datasetXval.topics(indXval, :);
end
if isfield(datasetLearn, 'cHistories')
  datasetLearn.cHistories = datasetLearn.cHistories(indLearn, :);
  datasetXval.cHistories = datasetXval.cHistories(indXval, :);
end
if isfield(datasetLearn, 'cTargets')
  datasetLearn.cTargets = datasetLearn.cTargets(indLearn, :);
  datasetXval.cTargets = datasetXval.cTargets(indXval, :);
end
if isfield(datasetLearn, 'hashmapBigrams')
  datasetXval.hashmapBigrams = datasetLearn.hashmapBigrams;
end
if isfield(datasetLearn, 'wordMatrix')
  datasetXval.wordMatrix = datasetLearn.wordMatrix;
end
