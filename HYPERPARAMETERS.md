# 🔧 Hyperparamètres des Algorithmes — Guide Explicatif

> **Comment lire ce document :**  
> Chaque algorithme liste ses hyperparamètres avec :  
> - La **valeur optimale** trouvée par GridSearchCV / RandomizedSearchCV  
> - Une explication simple en français de **ce que fait concrètement** ce paramètre  
> - Une analogie ou exemple pour mieux comprendre 💡

---

## 🔵 1. K-Nearest Neighbors (KNN)

> **Principe :** Pour prédire le type de peau d'un patient, le KNN regarde les **K patients les plus proches** dans l'espace des features et vote par majorité.  
> **Méthode d'optimisation :** GridSearchCV (validation croisée à 5 plis, 120 combinaisons)

---

### `n_neighbors = 17`

**Ce que ça fait :**  
Définit le nombre de **voisins les plus proches** à consulter pour faire une prédiction.

- Si `n_neighbors = 1` → le modèle copie exactement la classe du voisin le plus proche. Très sensible au bruit, risque de sur-apprentissage.
- Si `n_neighbors = 17` → le modèle fait voter 17 voisins. La classe majoritaire gagne. Plus robuste, plus général.

💡 **Analogie :** Imaginez que vous demandez l'avis de 17 dermatologues avant de décider du type de peau d'un patient. Plus on consulte de médecins, plus la décision est stable.

---

### `weights = 'distance'`

**Ce que ça fait :**  
Détermine comment les voisins contribuent au vote.

- `'uniform'` → tous les voisins ont le **même poids**, qu'ils soient proches ou loin.
- `'distance'` ✅ → les voisins **proches comptent plus** que les lointains (poids inversement proportionnel à la distance).

💡 **Analogie :** L'avis d'un dermatologue qui a examiné le patient de près (voisin proche) compte plus que l'avis de quelqu'un qui l'a vu de loin (voisin éloigné).

---

### `metric = 'euclidean'`

**Ce que ça fait :**  
Définit **comment mesurer la distance** entre deux patients dans l'espace des features.

- `'euclidean'` ✅ → distance en ligne droite dans l'espace multidimensionnel (formule de Pythagore généralisée). C'est la distance la plus naturelle et intuitive.
- `'manhattan'` → somme des différences absolues (comme se déplacer dans une grille de rues).
- `'minkowski'` → généralisation des deux précédentes.

💡 **Analogie :** Pour mesurer si deux patients se ressemblent, on calcule la "distance" entre leur âge, hydratation, sensibilité... comme on mesurerait la distance entre deux points sur une carte.

---
---

## 🌸 2. Decision Tree (Arbre de Décision)

> **Principe :** Construit un arbre de règles `SI ... ALORS ...` pour segmenter les patients en types de peau.  
> **Méthode d'optimisation :** Élagage CCP (Cost Complexity Pruning) + Pre-Pruning manuel

---

### `ccp_alpha = 0.000944`

**Ce que ça fait :**  
C'est le paramètre de **post-élagage** (Cost Complexity Pruning). Il contrôle à quel point on va couper les branches de l'arbre après qu'il ait été entièrement construit.

- `ccp_alpha = 0` → aucun élagage, l'arbre grandit jusqu'au bout. Résultat : 100% d'accuracy en train, 68.5% en test (sur-apprentissage total).
- `ccp_alpha = 0.000944` ✅ → on coupe les branches dont le gain d'information est trop faible. Résultat : arbre de profondeur 7, 79.3% en test.

💡 **Analogie :** Imaginez un arbre qui a grandi de manière incontrôlée avec des branches inutiles. `ccp_alpha` est le **sécateur** qui coupe les branches qui ne portent pas assez de fruits, pour que l'arbre reste sain et généralisable.

---

### `max_depth = 7`

**Ce que ça fait :**  
Limite la **profondeur maximale** de l'arbre, c'est-à-dire le nombre maximum de questions successives que l'arbre peut poser.

- `max_depth = None` → arbre illimité, qui peut poser 28 niveaux de questions → sur-apprentissage.
- `max_depth = 7` ✅ → au maximum 7 niveaux de questions, forçant l'arbre à ne retenir que les règles les plus générales.

💡 **Analogie :** C'est comme limiter un jeu de "20 questions" à seulement 7 questions. Vous êtes forcé de poser les questions les plus discriminantes en premier.

---

### `min_samples_split = 10`

**Ce que ça fait :**  
Nombre **minimum d'échantillons** qu'un nœud doit contenir pour qu'il puisse être divisé en sous-groupes.

- `min_samples_split = 2` → même 2 patients suffisent pour créer une nouvelle branche → trop de branches spécifiques.
- `min_samples_split = 10` ✅ → un groupe doit contenir au moins 10 patients pour être subdivisé → règles plus générales.

💡 **Analogie :** Un médecin ne crée pas un protocole de soin spécifique pour 2 patients seulement. Il faut au moins 10 cas similaires pour justifier une nouvelle règle clinique.

---

### `min_samples_leaf = 5`

**Ce que ça fait :**  
Nombre **minimum d'échantillons** qu'une feuille (nœud terminal) doit contenir.

- Si `min_samples_leaf = 1` → une feuille peut ne contenir qu'un seul patient → l'arbre mémorise des cas individuels.
- Si `min_samples_leaf = 5` ✅ → chaque décision finale doit être appuyée par au moins 5 patients → décisions plus robustes.

💡 **Analogie :** C'est le principe du **consensus médical** : un diagnostic ne peut être confirmé que s'il s'appuie sur au moins 5 cas observés, pas sur un cas isolé.

---
---

## 🌲 3. Random Forest (Forêt Aléatoire)

> **Principe :** Entraîne N arbres de décision en parallèle, chacun sur un sous-ensemble aléatoire des données et des features. La prédiction finale est le **vote majoritaire** de tous les arbres.  
> **Méthode d'optimisation :** GridSearchCV (validation croisée à 5 plis, 36 combinaisons)

---

### `n_estimators = 100`

**Ce que ça fait :**  
Le **nombre d'arbres** dans la forêt.

- Plus d'arbres → plus de stabilité, moins de variance → meilleure généralisation.
- Mais au-delà d'un certain seuil (~100-200 arbres), l'amélioration devient marginale et le temps de calcul augmente.
- `100` ✅ → bon compromis entre performance et efficacité computationnelle.

💡 **Analogie :** C'est comme sonder 100 dermatologues différents plutôt qu'un seul. Leur vote collectif est beaucoup plus fiable qu'une décision individuelle.

---

### `max_features = 'sqrt'`

**Ce que ça fait :**  
Le nombre de **features considérées à chaque nœud** lors de la construction d'un arbre.

- `'sqrt'` ✅ → à chaque division, l'arbre ne considère que √7 ≈ 2-3 features choisies aléatoirement (au lieu des 7 disponibles).
- Cela force chaque arbre à être **différent** des autres → diversité → réduction de la corrélation entre arbres → forêt plus robuste.

💡 **Analogie :** Chaque médecin dans le groupe ne regarde que quelques signes cliniques différents (hydratation, âge, sensibilité...). Ensemble, ils couvrent tout le spectre sans se copier mutuellement.

---

### `max_depth = None`

**Ce que ça fait :**  
Autorise les arbres de la forêt à **grandir sans limite de profondeur**.

- Contrairement au Decision Tree seul (dont l'élagage est crucial), dans une forêt, le sur-apprentissage de chaque arbre individuel est compensé par le vote collectif.
- `None` ✅ → chaque arbre est laissé grandir librement, mais la diversité due à `max_features` et `bootstrap` empêche la forêt de sur-apprendre.

---

### `min_samples_split = 2`

**Ce que ça fait :**  
Valeur minimale par défaut → permet aux arbres de se **spécialiser au maximum** sur leurs sous-ensembles de données respectifs.

Dans le contexte de la forêt, ce réglage agressif est acceptable car le bruit est moyenné sur 100 arbres.

---
---

## 🟠 4. Support Vector Machine (SVM)

> **Principe :** Trouve la **frontière de décision optimale** (hyperplan) qui maximise la marge entre les classes. Le noyau RBF projette les données dans un espace de plus haute dimension pour les rendre linéairement séparables.  
> **Méthode d'optimisation :** GridSearchCV (validation croisée à 3 plis, 12 combinaisons)

---

### `kernel = 'rbf'`

**Ce que ça fait :**  
La **fonction noyau** définit comment le SVM transforme l'espace des données pour trouver une frontière de décision.

| Noyau | Résultat sur ce dataset |
|:------|:------------------------|
| `'linear'` | Frontière droite — Accuracy = 77.4% |
| `'poly'` | Frontière polynomiale — Accuracy = 79.25% |
| `'rbf'` ✅ | Frontière gaussienne — **Accuracy = 79.25%** |
| `'sigmoid'` | Frontière sigmoïde — Accuracy = 62.8% |

- `'rbf'` (Radial Basis Function) ✅ → projette les données dans un espace de dimension infinie pour trouver des frontières complexes et courbes. Le meilleur choix pour des données non-linéairement séparables.

💡 **Analogie :** Imaginez que vous devez séparer des billes de couleurs mélangées sur une table. Le noyau RBF "soulève" certaines billes en 3D pour pouvoir passer un plan entre elles.

---

### `C = 1`

**Ce que ça fait :**  
Le paramètre de **régularisation** (ou de tolérance aux erreurs).

- `C` très petit (ex: 0.01) → le modèle accepte plus d'erreurs de classification mais produit une **marge large** → plus général.
- `C` très grand (ex: 100) → le modèle essaie de classer **tous** les points correctement → marge étroite → risque de sur-apprentissage.
- `C = 1` ✅ → bon équilibre entre généralisation et précision.

💡 **Analogie :** C'est le niveau de **tolérance à l'incertitude**. `C=1` signifie que le médecin accepte d'avoir quelques cas ambigus pour avoir un protocole de diagnostic plus généralisable.

---

### `gamma = 0.1`

**Ce que ça fait :**  
Contrôle la **portée d'influence** de chaque exemple d'entraînement dans le noyau RBF.

- `gamma` faible → chaque patient influence une large zone → frontière lisse, générale.
- `gamma` élevé → chaque patient influence une petite zone → frontière très courbée, risque de sur-apprentissage.
- `gamma = 0.1` ✅ → influence modérée, bon équilibre.
- `gamma = 'scale'` → valeur automatique = 1/(n_features × variance) ≈ calculée depuis les données.

💡 **Analogie :** C'est le **rayon d'attention** de chaque cas clinique. Avec `gamma=0.1`, chaque patient influence modérément les patients "proches" de lui, sans monopoliser la décision.

---
---

## 🟢 5. XGBoost (Extreme Gradient Boosting)

> **Principe :** Construit les arbres **séquentiellement**, chaque nouvel arbre corrigeant les erreurs du précédent. C'est du **boosting** : on amplifie l'apprentissage sur les cas difficiles.  
> **Méthode d'optimisation :** RandomizedSearchCV (40 combinaisons aléatoires, cv=3, durée : ~61 secondes)

---

### `max_depth = 7`

**Ce que ça fait :**  
Profondeur maximale de **chaque arbre individuel** dans la séquence de boosting.

- Arbres peu profonds (depth=3) → modèle plus simple, risque de sous-apprentissage.
- Arbres profonds (depth=7) ✅ → chaque arbre peut capturer des interactions complexes entre features, réduisant rapidement l'erreur résiduelle.

💡 **Analogie :** C'est la profondeur de l'analyse de chaque round de consultation. Plus l'arbre est profond, plus chaque consultant peut faire une analyse détaillée.

---

### `learning_rate = 0.3`

**Ce que ça fait :**  
Aussi appelé `eta`, c'est le **pas d'apprentissage** : contrôle de combien le modèle corrige ses erreurs à chaque étape.

- `learning_rate = 0.01` → très petits pas → apprentissage lent mais précis → nécessite beaucoup d'arbres (n_estimators élevé).
- `learning_rate = 0.3` ✅ → grands pas → apprentissage rapide → moins d'arbres nécessaires.

💡 **Analogie :** C'est la vitesse à laquelle un étudiant révise après chaque examen raté. Apprendre trop vite (0.3) peut mener à des erreurs, mais avec de la régularisation (gamma, reg_lambda), c'est efficace.

---

### `n_estimators = 100`

**Ce que ça fait :**  
Le **nombre de rounds de boosting**, c'est-à-dire le nombre d'arbres construits séquentiellement.

- Avec `learning_rate = 0.3`, 100 arbres suffisent pour converger vers une bonne solution.
- Si `learning_rate` était plus petit (0.01), il faudrait 300-500 arbres.

---

### `subsample = 0.8`

**Ce que ça fait :**  
La **proportion d'échantillons** utilisée pour entraîner chaque arbre (sans remise).

- `subsample = 1.0` → 100% des données utilisées → risque de sur-apprentissage.
- `subsample = 0.8` ✅ → chaque arbre ne voit que 80% des données → plus de diversité, moins de sur-apprentissage.

💡 **Analogie :** Chaque consultant dans la série ne consulte que 80% des dossiers patients, ce qui l'empêche de trop se concentrer sur des cas particuliers.

---

### `colsample_bytree = 0.5`

**Ce que ça fait :**  
La **proportion de features** sélectionnées aléatoirement pour construire chaque arbre.

- `0.5` ✅ → chaque arbre n'utilise que 50% des 7 features disponibles (≈ 3-4 features).
- Comme `max_features` dans Random Forest, cela force la **diversité entre arbres**.

---

### `gamma = 1.0`

**Ce que ça fait :**  
La **réduction minimale de perte** requise pour qu'un nœud soit divisé.

- `gamma = 0` → diviser librement, même pour un tout petit gain.
- `gamma = 1.0` ✅ → une division n'est effectuée que si elle réduit la perte d'au moins 1.0. Agit comme un **filtre anti-bruit** sur les divisions peu utiles.

💡 **Analogie :** C'est le seuil de rentabilité. On ne pose une question supplémentaire dans l'arbre que si elle améliore vraiment la précision de manière significative.

---

### `reg_lambda = 10`

**Ce que ça fait :**  
Régularisation **L2** (Ridge) sur les poids des feuilles des arbres.

- Plus `reg_lambda` est élevé → poids des feuilles forcés à rester petits → modèle plus lisse → moins de sur-apprentissage.
- `reg_lambda = 10` ✅ → régularisation forte, efficace sur ce dataset bruité.

---

### `reg_alpha = 0.1`

**Ce que ça fait :**  
Régularisation **L1** (Lasso) sur les poids des feuilles.

- L1 peut annuler complètement des poids (mettre à zéro) → favorise la **parcimonie**.
- `reg_alpha = 0.1` ✅ → légère régularisation L1, complémentaire à L2.

---

### `min_child_weight = 3`

**Ce que ça fait :**  
La **somme minimale des poids d'instance** requise dans un nœud feuille. Similaire à `min_samples_leaf` du Decision Tree.

- `min_child_weight = 1` → feuilles très petites autorisées → sur-apprentissage.
- `min_child_weight = 3` ✅ → chaque feuille doit contenir suffisamment "d'information" pour justifier une décision.

---
---

## 📊 Tableau Récapitulatif Global

| Param. \ Algo | KNN | Decision Tree | Random Forest | SVM | XGBoost |
|:--------------|:----|:--------------|:--------------|:----|:--------|
| Voisins / Arbres | `k=17` | — | `100 trees` | — | `100 rounds` |
| Profondeur | — | `depth=7` | `depth=None` | — | `depth=7` |
| Régularisation | `weights=dist.` | `ccp=0.000944` | `max_feat=sqrt` | `C=1` | `γ=1, λ=10, α=0.1` |
| Noyau / Distance | `metric=eucl.` | — | — | `kernel=rbf` | — |
| Pas d'apprentissage | — | — | — | `gamma=0.1` | `lr=0.3` |
| Sous-échantillonnage | — | `min_leaf=5` | — | — | `sub=0.8, col=0.5` |
| **Accuracy** | **77.6%** | **79.3%** | **~81%** | **79.25%** | **78.95%** |

---

## 💡 Règle d'or à retenir

> **Tous ces hyperparamètres servent le même objectif : trouver le bon équilibre entre la mémorisation (sur-apprentissage) et la généralisation.**
>
> - Trop de flexibilité → le modèle mémorise les données d'entraînement mais échoue sur les nouvelles.  
> - Trop de contraintes → le modèle est trop simple pour capturer les vraies tendances.
>
> C'est le **compromis biais-variance**, et GridSearchCV / RandomizedSearchCV automatisent la recherche de ce point d'équilibre optimal.

---

*DeepSkyn — Projet Académique 4TWIN, Semestre 2*
