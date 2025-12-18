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
