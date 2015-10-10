function [sim, resnik] = NLP_WordNetSimilarity(word1, word2, resnik)

word1 = lower(word1);
word2 = lower(word2);

if (nargin < 3)
  resnik = InitializeWordNet();
end
  
sim = resnik.max(word1, word2, 'n');


% -------------------------------------------------------------------------
function resnik = InitializeWordNet(path_wordnet)

if (nargin < 1)
  path_wordnet = ...
    '/Users/piotr/Documents/Projets/libs/WordNet-Similarity-Java';
end

javaaddpath([path_wordnet '/' 'edu.sussex.nlp.jws.beta.11.jar']);
javaaddpath([path_wordnet '/' 'edu.mit.jwi_2.1.4.jar']);
import edu.sussex.nlp.jws.JWS;
import edu.mit.jwi_2.1.4.*
jws = JWS(path_wordnet, '3.0');
resnik = jws.getResnik();


% url1 = 'http://marimba.d.umn.edu/cgi-bin/similarity/similarity.cgi?';
% url2 = 'word1=';
% url3 = '&senses1=all&word2=';
% url4 = '&senses2=all&measure=path&rootnode=yes';
% 
% n_tries = 1;
% while (n_tries < 10)
%   try
%     res = urlread([url1 url2 word1 url3 word2 url4]);
%     break;
%   catch
%     fprintf(1, 'Waiting for %d seconds\n', n_tries);
%     pause(n_tries);
%     n_tries = n_tries + 1;
%     dummy = urlread('http://www.google.com');
%   end
% end
% if (n_tries >= 10)
%   error('Could not download after 10 tries...');
% end
% 
% 
% try
%   pos = regexp(res, '<p class="results">The relatedness of ');
%   res = res(pos:end);
%   pos = regexp(res, 'using path is ');
%   res = res(pos + [14:50]);
%   sim = sscanf(res, '%g');
% catch
%   sim = 0;
% end
