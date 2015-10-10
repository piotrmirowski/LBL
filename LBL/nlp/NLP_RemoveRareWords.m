% NLP_RemoveRareWords  Remove rare words from the datasets
%
% Syntax:
%   [datasetTrain, datasetTest] = ...
%     NLP_RemoveRareWords(datasetTrain, datasetTest, n_rare, ...
%     tok_bos, tok_rare, tok_unk)
% Inputs:
%   datasetTrain: struct containing the train dataset
%   datasetTest:  struct containing the test dataset
%   n_rare:       number of word occurrences considered as "rare"
%   tok_bos:      (character string), word in vocabulary used to begin
%                 a sentence
%   tok_rare:     (character string), word in vocabulary used for rare word
%   tok_unk:      (character string), word in vocabulary used for unknown
% Outputs:
%   datasetTrain: struct containing the train dataset
%   datasetTest:  struct containing the test dataset

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

function [datasetTrain, datasetTest] = ...
  NLP_RemoveRareWords(datasetTrain, datasetTest, n_rare, ...
  tok_bos, tok_rare, tok_unk)

% Get the word counts and compute the unigram at the same time
if ~isfield(datasetTrain, 'unigram');
  datasetTrain = LBL_Statistics_Unigram(datasetTrain);
end
n_words_train = length(datasetTrain.wTargets);
wordCountsTrain = round(datasetTrain.unigram * n_words_train);


% Hack not to remove the beginning-of-sentence tag BOS
[tag_bos, found] = NLP_FindWord(tok_bos, datasetTrain.vocabulary);
if ~found, error('No bos'); end
wordCountsTrain(tag_bos) = datasetTrain.lines(end);
fprintf(1, 'There are %d %s tags in the train data (# lines)...\n', ...
  wordCountsTrain(tag_bos), tok_bos);


% Hack not to remove the unknown tag
[tag_unk, found] = NLP_FindWord(tok_unk, datasetTrain.vocabulary);
if ~found, error('No unk'); end
wordCountsTrain(tag_unk) = sum(datasetTest.wTargets == tag_unk);
fprintf(1, 'There are %d %s tags in the test targets...\n', ...
  wordCountsTrain(tag_unk), tok_unk);


% Replace the rare words by the token <tok_rare> (if relevant)
if (n_rare > 0)
  [tag_rare, found] = NLP_FindWord(tok_rare, datasetTrain.vocabulary);
  if ~found, error('No %s', tok_rare); end

  % Replace rare words
  indRare = find((wordCountsTrain > 0) & (wordCountsTrain <= n_rare));
  fprintf(1, 'Replacing rare words in training data...\n');
  datasetTrain = NLP_ReMapWords(datasetTrain, indRare, tag_rare);
  fprintf(1, 'Replacing rare words in test data...\n');
  datasetTest = NLP_ReMapWords(datasetTest, indRare, tag_rare);
  
  % Update the word counts
  n = sum(wordCountsTrain(indRare));
  wordCountsTrain(indRare) = 0;
  wordCountsTrain(tag_rare) = wordCountsTrain(tag_rare) + n;
end
  

% Select only words that appear in the (new) dataset
indMapWords = find(wordCountsTrain > 0);
fprintf(1, 'Removing unused words in training data...\n');
datasetTrain = NLP_ReMapWords(datasetTrain, indMapWords);
datasetTrain.vocabulary = datasetTrain.vocabulary(indMapWords);
fprintf(1, 'Removing unused words in test data...\n');
datasetTest = NLP_ReMapWords(datasetTest, indMapWords);
datasetTest.vocabulary = datasetTest.vocabulary(indMapWords);
wordCountsTrain = wordCountsTrain(indMapWords);


% Find the unk tag again
[tag_unk, found] = NLP_FindWord(tok_unk, datasetTrain.vocabulary);
datasetTrain.tag_unk = tag_unk;
datasetTest.tag_unk = tag_unk;
wordCountsTrain(tag_unk) = 0;


% Hack not to remove the beginning of the sentence
[tag_bos, found] = NLP_FindWord(tok_bos, datasetTrain.vocabulary);
wordCountsTrain(tag_bos) = 0;
datasetTrain.unigram = wordCountsTrain / n_words_train;
datasetTrain.tag_bos = tag_bos;
datasetTest.tag_bos = tag_bos;

[tag_rare, found] = NLP_FindWord(tok_rare, datasetTrain.vocabulary);
datasetTrain.tag_rare = tag_rare;
datasetTest.tag_rare = tag_rare;

