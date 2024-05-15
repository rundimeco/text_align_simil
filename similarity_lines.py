import glob
from sklearn.feature_extraction.text import CountVectorizer
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

corpus = {}

#Note pour plus tard : sur un Path on peut faire .glob

for chemin in glob.glob("data/noisy/corrige/*.txt"):
    name = Path(chemin).name
    with open(chemin, encoding="utf-8") as f:
      texte = f.readlines()
      corpus[name] = texte

brut = {}
for chemin in glob.glob("data/noisy/brut/*.txt"):
    name = Path(chemin).name
    with open(chemin, encoding="utf-8") as f:
      texte = f.readlines()
      brut[name] = texte

for name, lines_corrigee in corpus.items():
  w = open(f"alignement_{name}.txt", "w")
  w.write("Similarité\tLigne Corrigée\tLigne Brute\n")
  vectorizer = CountVectorizer(analyzer="char", ngram_range= (2,3))
  NB_corrigees = len(lines_corrigee)
  lines_brut = brut[name]
  X = vectorizer.fit_transform(lines_corrigee+lines_brut)
  matrix = cosine_similarity(X.toarray())
  for i in range(NB_corrigees):
      sim_bruit = matrix[i][NB_corrigees:]
      maxi = max(sim_bruit)
      pos = [sim_pos for sim_pos, sim_line in enumerate(sim_bruit) if sim_line == maxi]
      if len(pos)==1:
        maxi = round(maxi, 4)
        corr = re.sub("\r|\n"," ", lines_corrigee[i])
        br = re.sub("\r|\n", "", lines_brut[pos[0]])
        w.write(f"{maxi}\t{corr}\t{br}\n")
  w.close()
