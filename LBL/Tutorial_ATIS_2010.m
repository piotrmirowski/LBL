% Add paths
addpath('util');
addpath('lbl');
addpath('nlp');


% Paths to data and results (might need to be modified)
path_data = '../data/ATIS/';
path_results = '../results/ATIS/';

% -------------------------------------------------------------------------
% 1) Import the text files of the corpus into Matlab structures
% -------------------------------------------------------------------------

% Files containing the vocabulary and POS tags
filename_vocab = 'ATIS.train.vocab';
filename_pos = 'ATIS.pos';

% Filename for the Matlab data
filename_data = 'ATIS.5grams.mat';

% Can we just load the Matlab file or do we need to regenerate it?
load_data = exist([path_data filename_data], 'file');

if (~load_data)
  % Import the vocabulary and POS tags
  % These files need to be generated using shell scripts
  fprintf(1, 'Reading vocabulary and POS files %s and %s...\n', ...
    filename_vocab, filename_pos);
  vocabulary = importdata([path_data filename_vocab]);
  posTags = importdata([path_data filename_pos]);

  % The above command might fail if some words have special punctation.
  % An alternative way to import the vocabulary is by using function
  % NLP_ReadVocabulary, which scans the text file line by line
  % Or, simply read the shell script-generated vocabulary file line by line.
  
  % Sort the vocabulary and POS tags (needed for a proper corpus import, as
  % we use a tree-like recursive search in the vocabulary and POS lists)
  vocabulary = sort(vocabulary);
  posTags = sort(posTags);
end

if (~load_data)
  % Filenames of the training and test data
  filename_train = 'ATIS.train.pos.data';
  filename_test = 'ATIS.test.pos.data';

  % Import the training dataset as 5-grams
  % (i.e. 4-gram history and prediction), with vocabulary and POS tags,
  % skipping lines with OOV words (but there are no OOV words in train data)
  fprintf(1, 'Importing training text file %s...\n', filename_train);
  datasetTrain_5 = ...
    NLP_ReadFileWithPOS([path_data filename_train], 5, ...
    vocabulary, posTags, 1);

  % N.B.: the ATIS should look like the following, with <word>//<POS_tag>
  % sequences, one sentence per line:
  % A//DT A//NNP THIRTY//NN SEVEN//NN
  % A//DT BREAKFAST//NN FLIGHT//NN FROM//IN DENVER//NN TO//TO SAN//NN FRANCISCO//NN PLEASE//NN
  % A//DT CITY//NN TO//TO RELAX//NNS

  % The dataset structure contains the vocabulary.
  % After text import, the vocabulary might be augmented by '{bos}'
  % (beginning of sentence) and '{unk}' (unknown) tags
  % Just to keep things clean, retrieve the augmented vocabulary.
  vocabulary = datasetTrain_5.vocabulary;

  % Import the test dataset as 5-grams
  % (i.e. 4-gram history and prediction), with vocabulary and POS tags,
  % skipping lines with OOV words (and there are some OOV words in test data)
  fprintf(1, 'Importing test text file %s...\n', filename_test);
  datasetTest_5 = ...
    NLP_ReadFileWithPOS([path_data filename_test], 5, ...
    vocabulary, posTags, 1);
end


% -------------------------------------------------------------------------
% 2) Save everything into a Matlab file, for faster retrieval
% -------------------------------------------------------------------------

% Filename for the Matlab data
filename_data = 'ATIS.5grams.mat';

% Save the dataset into a Matlab file
if (~load_data)
  save([path_data filename_data], 'datasetTrain_5', 'datasetTest_5', ...
    'vocabulary', 'posTags');
else
  load([path_data filename_data]);
end


% -------------------------------------------------------------------------
% 3) Define the parameters
% -------------------------------------------------------------------------

% Define an LBLN with 103-dimensional hidden variables (|Z|=103):
%   |Z_W|=100 for words (with |W|=1312 words in the training corpus)
%   |Z_X|=3 for tags (with |X|=30 POS tags in the training corpus)
% We use 4-gram histories (since these are 5-grams).
% We also use 200-dimensional hidden nodes in the nonlinear component.
% Default parameters are explained in LBL_Params_Init
% params = LBL_Params_Init(1312, 103, 4, 200, 30, 100);
params = LBL_Params_Init(1312, 100, 4, 0, 0, 100);

% Change the number of epochs to 100
params.n_epochs = 100;

% Set the initialization of word representation to random
params.init_r = 'inv_sqrt';

% Evaluate the model on training data after each epoch
params.eval_ppx_train = 1;
disp(params);


% -------------------------------------------------------------------------
% 4) Train the language model
% -------------------------------------------------------------------------

% Split the training data into 95% learning set and 5% cross-validation set
[datasetLearn_5, datasetXval_5] = LBL_Dataset_Split(datasetTrain_5, 0.95);

% Launch a learning session with the learning, x-validation and test set,
% for 5 epochs (see parameters) or until "convergence" (the model stores
% the parameters associated to the lowest x-validation perplexity), and
% stops training when the x-validation perplexity increases 5 times
% consecutively.
LBL_Trainer(datasetLearn_5, datasetXval_5, datasetTest_5, params);

% During training, after each epoch, a .mat and a .txt files are updated,
% containing all the results (or summary statistics of them).

% The meters can be accessed using the following global variables
global METER_TRAIN
global METER_TEST
global METER_XVAL

% The model can be access using the following variable
global MODEL

% Of course, these variables, as well as the parameters,
% are also saved to a file, and parameters and results
% are printed into a text file, after each epoch

% Plot the learning performance using the meters
figure;
hold on;
plot(METER_TRAIN.ppx_e, 'b');
plot(METER_XVAL.ppx_e, 'g');
plot(METER_TEST.ppx_e, 'r');
legend('Train', 'X-validation', 'Test');
xlabel('Epochs');
ylabel('Perplexity');
title(sprintf('Learning performance for run %s', params.filename), ...
  'Interpreter', 'none')
grid on


% -------------------------------------------------------------------------
% 5) Play with the language model
% -------------------------------------------------------------------------

% File where the word embeddings are written to
filename_embed = [params.filename '_Embed.txt'];

% Print the word embeddings to a file, using 12-neighborhoods,
% and for the first 1310 words, 
fid = fopen(filename_embed, 'w');
LBL_Evaluate_Embedding(MODEL, fid, 1:1312, 12);
fclose(fid);

% File where the word predictions are written to
filename_embed = [params.filename '_Pred.txt'];

% Print the 5 top predictions for each n-gram in the test dataset
% This also plots a pie chart of the rank of the correct predicted word
% (is it the first one? is it in the top 5? in the top 100?, etc...)
% This also produces the ranks of the correct word for each word prediction
figure;
fid = fopen(filename_embed, 'w');
ranks = LBL_Evaluate_Rank(datasetTest_5, MODEL, params, fid);
fclose(fid);

% In the above, if one wants to print on the screen, just use fid=1
