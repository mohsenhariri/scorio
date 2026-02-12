# Examples

## Bayesian Evaluation

The `bayes` function implements the Bayes@N framework. It takes a matrix of outcomes `R`, a weight vector `w`, and optionally a matrix of prior outcomes `R0`.

```julia
using Scorio

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = [0 1 2 2 1;
     1 1 0 2 2]

# Rubric weights w: length C+1
# Here: 0=incorrect(0.0), 1=partial(0.5), 2=correct(1.0)
w = [0.0, 0.5, 1.0]

# Optional prior outcomes R0: shape (M, D)
R0 = [0 2;
      1 2]

# Bayesian evaluation with prior
mu, sigma = bayes(R, w, R0)
println("μ = $mu, σ = $sigma")
```

## Pass@k Evaluation

For binary outcomes (correct/incorrect), you can use `pass_at_k` and `pass_hat_k`.

```julia
using Scorio

# Binary outcomes R: shape (M, N)
R = [0 1 1 0 1;
     1 1 0 1 1]

# Pass@k: Probability that at least one of k samples is correct
pk = pass_at_k(R, 1)
println("Pass@1: $pk")

# Pass^k (Pass-hat@k): Probability that all k samples are correct
phk = pass_hat_k(R, 2)
println("Pass^2: $phk")
```

## Ranking

You can compute competition ranks from scores using `competition_ranks_from_scores`.

```julia
using Scorio

scores = [0.95, 0.87, 0.87, 0.72, 0.65]
ranks = competition_ranks_from_scores(scores)
# Returns: [1, 2, 2, 4, 5]
println(ranks)
```

## Ranking APIs (Response Tensor)

Ranking methods operate on a response tensor `R` with shape `(L, M, N)`:
- `L`: number of models
- `M`: number of questions
- `N`: number of trials per question

```julia
using Scorio

# R[l, m, n] = 1 if model l solves question m on trial n, else 0
R = [
    1 1 0 1 0;
    1 0 0 1 0;
    0 1 0 1 1;
    0 0 0 1 0
]

# Promote to (L, M, N=1) automatically when needed.
# For multi-trial ranking, pass a 3D tensor explicitly:
R3 = reshape(R, 4, 5, 1)
```

### Eval-Metric Ranking

```julia
using Scorio

# Mean-accuracy ranking
ranks_mean, scores_mean = Scorio.mean(R3; return_scores=true)
println("mean ranks  = ", ranks_mean)
println("mean scores = ", scores_mean)

# Stability-style ranking with generalized pass metric
ranks_gpass, scores_gpass = g_pass_at_k_tau(R3, 1, 1.0; return_scores=true)
println("g_pass ranks  = ", ranks_gpass)
println("g_pass scores = ", scores_gpass)
```

### Pairwise Rating Methods

```julia
using Scorio

ranks_elo, ratings_elo = elo(R3; K=16.0, return_scores=true)
println("Elo ranks   = ", ranks_elo)
println("Elo ratings = ", ratings_elo)

ranks_ts, scores_ts = trueskill(R3; return_scores=true)
println("TrueSkill ranks = ", ranks_ts)
println("TrueSkill mu    = ", scores_ts)

ranks_glicko, ratings_glicko, rd_glicko = glicko(R3; return_deviation=true)
println("Glicko ranks   = ", ranks_glicko)
println("Glicko ratings = ", ratings_glicko)
println("Glicko RD      = ", rd_glicko)
```

### Bradley-Terry Family

```julia
using Scorio

ranks_bt, scores_bt = bradley_terry(R3; return_scores=true)
println("BT ranks  = ", ranks_bt)
println("BT scores = ", scores_bt)

ranks_bt_map, scores_bt_map = bradley_terry_map(R3; prior=1.0, return_scores=true)
println("BT-MAP ranks  = ", ranks_bt_map)
println("BT-MAP scores = ", scores_bt_map)
```

### Voting-Based Ranking

```julia
using Scorio

ranks_borda, scores_borda = borda(R3; return_scores=true)
println("Borda ranks  = ", ranks_borda)
println("Borda scores = ", scores_borda)

ranks_copeland, scores_copeland = copeland(R3; return_scores=true)
println("Copeland ranks  = ", ranks_copeland)
println("Copeland scores = ", scores_copeland)
```

### IRT-Based Ranking

```julia
using Scorio

ranks_rasch, scores_rasch = rasch(R3; return_scores=true)
println("Rasch ranks  = ", ranks_rasch)
println("Rasch theta  = ", scores_rasch)

ranks_2pl, scores_2pl = rasch_2pl(R3; return_scores=true)
println("2PL ranks  = ", ranks_2pl)
println("2PL theta  = ", scores_2pl)
```

### Graph and Listwise Ranking

```julia
using Scorio

ranks_pr, scores_pr = pagerank(R3; return_scores=true)
println("PageRank ranks  = ", ranks_pr)
println("PageRank scores = ", scores_pr)

ranks_rc, scores_rc = rank_centrality(R3; return_scores=true)
println("RankCentrality ranks  = ", ranks_rc)
println("RankCentrality scores = ", scores_rc)

ranks_pl, scores_pl = plackett_luce(R3; return_scores=true)
println("Plackett-Luce ranks  = ", ranks_pl)
println("Plackett-Luce scores = ", scores_pl)
```
