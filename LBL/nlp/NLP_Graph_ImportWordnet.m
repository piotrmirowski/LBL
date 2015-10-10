% NLP_Graph_ImportWordnet  Import the WordNet::Similarity (Resnik) graph
%
% Syntax:
%   graph = ...
%     NLP_Graph_ImportWordnet(file_graph, vocabularyGraph, 
%                             vocabulary, min_sim)
% Inputs:
%   file_graph:      char string with filename for the graph file,
%                    whose 0-indexed format is as following:
%                    <source_word_idx> [<target_word_idx>:<score> ]*
%                    e.g. for the ATIS dataset:
% 0 266:11.766 343:11.766 162:10.667 1098:10.667 1203:9.974 107:9.686 ...
% 1
% 2
% 3 319:8.300 18:7.161 1063:7.161 1287:6.020 215:5.575 299:5.575 ...
% ...
% 1305
% 1306
% 1307 0:8.993 46:8.993 107:8.993 162:8.993 266:8.993 282:8.993 ...
% 1308 103:7.591 52:6.906 1166:6.906 748:6.769 752:6.769 ...
% 1309 284:6.590 365:6.590 622:6.590 820:6.590 835:6.590 1077:6.590 ...
%   vocabularyGraph: cell array of char strings for words, used in the
%                    graph construction, if different from <vocabulary>
%   vocabulary:      cell array of char strings for words, used in the 
%                    text data reading, if different from <vocabularyGraph>
%   min_sim:         minimum similarity score to be kept
% Outputs:
%   graph:           sparse matrix of size <dim_w> x <dim_w>

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

function graph = ...
  NLP_Graph_ImportWordnet(file_graph, vocabularyGraph, vocabulary, min_sim)

% Match the vocabulary of the graph with the dataset vocabulary
map = NLP_MatchVocabularies(vocabularyGraph, vocabulary);

% Allocate graph
dim_w = length(vocabulary);
graph = zeros(dim_w, dim_w);


% Import the similarity graph from <filename>
fprintf(1, 'Loading the similarity graph...\n');
fid = fopen(file_graph, 'r');
while (1)
  line = fgetl(fid);
  if ~ischar(line), break; end

  % Get the index of word w.r.t. which similarity is computed
  [i_str, line] = strtok(line, ' ');
  i = sscanf(i_str, '%d', 1) + 1;

  % Retrieve the similar words and their indexes
  vals = sscanf(line,'%d:%g', Inf);

  % Fill the graph with similarity values above a threshold
  js = vals(1:2:end) + 1;
  vals = vals(2:2:end);
  ind = (vals > min_sim);
  graph(map(i), map(js(ind))) = vals(ind);

  if (mod(i, 100) == 0)
    fprintf(1, '.');
  end
end
fclose(fid);
fprintf(1, '\n');


% Stem the vocabulary
fprintf(1, 'Stemming the vocabulary...\n');
for k = 1:dim_w
  try
    stems{k} = porterStemmer(lower(vocabulary{k}));
  catch
    fprintf(1, 'Could not stem %s\n', stems{k});
  end
  if (mod(k, 100) == 0)
    fprintf(1, '.');
  end
end
fprintf(1, '\n');


% Merge the words that have the same stem
fprintf(1, 'Merging the graphs of identical stems...\n');
i = 1;
k = 0;
while (i < dim_w)
  stem_i = stems{i};
  ind = [];
  j = i + 1;
  cond = 1;
  while (cond)
    stem_j = stems{j};
    if isequal(stem_i, stem_j)
      ind = [ind j];
    end
    j = j + 1;
    if ((j > dim_w) || (stem_j(1) ~= stem_i(1)))
      cond = 0;
    end
  end
  
  if isempty(ind)
    i = i + 1;
  else
    ind = [i ind];

    % Skip ahead
    i = ind(end) + 1;

    % Merge the graphs
    graph_ind = max(graph(ind, :));
    for j = ind
      graph(j, :) = graph_ind;
      graph(j, setdiff(ind, j)) = 11;
    end
    
    % Trace
    if (mod(k, 100) == 0)
      fprintf(1, '.');
    end
    k = k + 1;
  end
end
fprintf(1, '\n');


