function map = NLP_MatchVocabularies(voca1, voca2)

n_words1 = length(voca1);
n_words2 = length(voca2);
n_words = max(n_words1, n_words2);
map = zeros(n_words, 1);

j = 1;
for i = 1:n_words1
  voca1i = voca1{i};
  if ((j > n_words2) || ((j <= n_words2) && ~isequal(voca1i, voca2{j})))
    found = 0;
    for j = 1:n_words2
      if isequal(voca1i, voca2{j})
        found = 1;
        break;
      end
    end
    if ~found
      fprintf(1, 'Could not find a match for word %s from voca1\n', ...
        voca1i);
      j = 0;
    end
  end
  map(i) = j;
  j = j + 1;
end
