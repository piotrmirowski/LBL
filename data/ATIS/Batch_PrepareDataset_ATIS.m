% Read the vocabulary
vocabularyTrain = NLP_ReadVocabulary('ATIS.train.data');
vocabularyTest = NLP_ReadVocabulary('ATIS.test.data');

% Add the blank sign (beginning or end of sentence)
vocabulary = union(vocabularyTrain, union(vocabularyTest, {'<S>', '</S>'}));

% Extract the n-grams
for n = 2:6
  datasetTrain{n} = NLP_ReadFile('ATIS.train.data', n, vocabulary);
  datasetTest{n} = NLP_ReadFile('ATIS.test.data', n, vocabulary);
end
