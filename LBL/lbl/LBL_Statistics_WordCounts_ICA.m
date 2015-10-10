% LBL_Statistics_WordCounts_ICA  Compute ICA topics from word counts
%
% Syntax:
%   [thetaTrain, thetaTest] = ...
%     LBL_Statistics_WordCounts_ICA(wordCountsTrain, wordCountsTest, n_k)
% Inputs:
%   wordCountsTrain: matrix of size <dim_w> x <n_docs_1> of word counts
%   wordCountsTest:  matrix of size <dim_w> x <n_docs_2> of word counts
%   n_k:             number of independent components
% Outputs:
%   thetaTrain: matrix of size <n_k> x <n_docs_1> of document embeddings
%   thetaTest:  matrix of size <n_k> x <n_docs_2> of document embeddings

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

function [thetaTrain, thetaTest] = ...
  LBL_Statistics_WordCounts_ICA(wordCountsTrain, wordCountsTest, n_k)


% Compute the TF-IDF scores for each word
tfidfTrain = IE_ComputeTFIDF(wordCountsTrain);
tfidfTest = IE_ComputeTFIDF(wordCountsTest);


% Compute the ICA with dimensionality reduction to <n_k> components
[thetaTrain, wDecoder, wEncoder] = ...
  fastica(tfidfTrain, 'g', 'tanh', 'lastEig', dim_z, 'numOfIC', n_k);

% Extract the ICA components from test data
thetaTest = wEncoder * tfidfTest;
