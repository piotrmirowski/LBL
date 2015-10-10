function [datasetTrain, datasetTest] = ...
  NLP_RemoveRareWords_Patch(datasetTrain, datasetTest, word_replace)


% Useful tags
[tag_bos, found] = NLP_FindWord('bos', datasetTrain.vocabulary);
if ~found, error('No bos'); end
[tag_replace, found] = NLP_FindWord(word_replace, datasetTrain.vocabulary);
if ~found, error('No %s', word_replace); end

% Get the word counts and compute the unigram at the same time
if ~isfield(datasetTrain, 'unigram');
  datasetTrain = LBL_Statistics_Unigram(datasetTrain);
end
n_words_train = length(datasetTrain.wTargets);
wordCountsTrain = round(datasetTrain.unigram * n_words_train);

% Hack not to remove the beginning-of-sentence tag BOS
wordCountsTrain(tag_bos) = datasetTrain.sentences(end);


% In the test dataset, replace words that do not appear in the unigram
% by word <word_replace>
indNot = find(wordCountsTrain == 0);
fprintf(1, 'Removing words in test data that should not be there...\n');
dim_not = length(indNot);
if (dim_not > 0)
  for j = 1:dim_not
    k = indNot(j);
    indTargets = (datasetTest.wTargets == k);
    indHistories = (datasetTest.wHistories == k);
    datasetTest.wTargets(indTargets) = tag_replace;
    datasetTest.wHistories(indHistories) = tag_replace;
    
    if (mod(j, 100) == 0)
      fprintf(1, 'Removed %5d/%5d words...\n', j, dim_not);
    end
  end
end
