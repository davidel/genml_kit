# Reinforcement Learning — A Self-Contained Tutorial

This document teaches, from the ground up, the mathematics of reinforcement
learning (RL): how an agent that only ever sees *states*, takes *actions*,
and receives *rewards* can be trained into a policy that maximises
cumulative reward.

We assume only what an engineering undergraduate knows: calculus, linear
algebra, basic probability, and a working familiarity with automatic
differentiation of a parameterised function (e.g. a neural network trained
with `backward()`).  No prior exposure to reinforcement learning is assumed.

The document is written in a fixed pedagogical order: definitions first,
then derivations, then *proofs* where the claim is a theorem.  Blind trust
is never required; every formula that matters is derived here.

---

## How To Read This Document

- **Read order.**  The chapters are ordered so that each one builds on the
  previous.  Part 1 defines the *problem* (Markov decision process, returns,
  value functions, Bellman equations) and proves the two structural results
  everything else leans on: the Bellman recursion and the contraction
  property that guarantees a unique solution.  Part 2 derives the *tabular*
  learning rules (TD, Q-learning) from the Bellman equations and explains
  why they are the correct thing to do when the model is unknown.  Part 3
  moves to *function approximation*: DQN, replay, target networks, Double
  DQN and Dueling DQN.  Part 4 derives the *policy-gradient theorem*, the
  actor-critic reduction, GAE and the PPO clipped surrogate.  Part 5 derives
  *maximum-entropy* RL and Soft Actor-Critic.
- **What is derived vs. what is asserted.**  Statements labelled
  **Proof** or **Derivation** are shown in full.  Statements labelled
  *Claim* are true but their proofs are standard textbook material; we give
  the precise reference in Appendix C so nothing is taken on faith silently.
- **Notation.**  Tables (Appendix A) spell out every symbol at first use.
  Vectors are lowercase bold $`\boldsymbol{x}`$; scalars are lowercase
  italic $`x`$; random variables are uppercase $`S_t`$; matrices are
  uppercase italic $`M`$; probability mass/density is $`p`$; expectations
  $`\mathbb{E}`$; policy $`\pi`$; discount $`\gamma`$.

---

## Contents

- [0. The Mathematical Toolkit](#0-the-mathematical-toolkit)
- **Part 1 — The Problem: Sequential Decisions Under Uncertainty**
  - [1. What Problem Are We Solving?](#1-what-problem-are-we-solving)
  - [2. The Markov Decision Process, Taught From Scratch](#2-the-markov-decision-process-taught-from-scratch)
  - [3. Value Functions And The Bellman Equations](#3-value-functions-and-the-bellman-equations)
- **Part 2 — Learning From Experience: Temporal-Difference Methods**
  - [4. Monte Carlo, Bootstrapping, And The TD Error](#4-monte-carlo-bootstrapping-and-the-td-error)
  - [5. Q-Learning: Off-Policy Control By A Single-Step Bootstrap](#5-q-learning-off-policy-control-by-a-single-step-bootstrap)
- **Part 3 — Deep Q-Learning**
  - [6. Function Approximation, Replay, And Target Networks](#6-function-approximation-replay-and-target-networks)
  - [7. Double DQN: Fixing Overestimation](#7-double-dqn-fixing-overestimation)
  - [8. Dueling DQN: The Advantage Decomposition](#8-dueling-dqn-the-advantage-decomposition)
- **Part 4 — Policy Gradients And PPO**
  - [9. The Policy Gradient Theorem](#9-the-policy-gradient-theorem)
  - [10. Actor-Critic And Generalized Advantage Estimation](#10-actor-critic-and-generalized-advantage-estimation)
  - [11. Trust Regions And The Clipped Surrogate (PPO)](#11-trust-regions-and-the-clipped-surrogate-ppo)
- **Part 5 — Maximum-Entropy RL And Soft Actor-Critic**
  - [12. Soft Value Functions And Soft Bellman Equations](#12-soft-value-functions-and-soft-bellman-equations)
  - [13. Soft Actor-Critic](#13-soft-actor-critic)
- **Appendices**
  - [Appendix A: Symbol Table](#appendix-a-symbol-table)
  - [Appendix B: Failure Modes And Shortcuts](#appendix-b-failure-modes-and-shortcuts)
  - [Appendix C: Reading List](#appendix-c-reading-list)

---

## 0. The Mathematical Toolkit

This chapter collects, with proofs, the few analytic tools the rest of the
document uses repeatedly.  A reader fluent in these can read the derivations
in Parts 1–5 line by line; a reader who has not seen them should work through
this chapter once, slowly.

### 0.1 The Law of Total Expectation

**Theorem (law of total expectation).**  Let $`X`$ be a random variable and
$`Y`$ a random variable (or vector).  If $`\mathbb{E}[|X|] < \infty`$, then

$$
\large
\mathbb{E}[X] = \mathbb{E}\\!\left[\\, \mathbb{E}[X \mid Y] \\,\right].
$$

**Proof.**  Take the case where $`Y`$ is discrete with probability mass
$`p_Y(y)`$.  By the definition of conditional expectation,

$$
\large
\mathbb{E}[X \mid Y = y] = \sum_x x\\, p(x \mid y),
\qquad
p(x \mid y) = \frac{p(x, y)}{p_Y(y)}.
$$

Rearranging the double sum,

$$
\large
\begin{aligned}
\mathbb{E}\\!\left[\\, \mathbb{E}[X \mid Y] \\,\right]
&= \sum_y p_Y(y) \sum_x x\\, p(x \mid y) \\
&= \sum_x x \sum_y p_Y(y)\\, p(x \mid y)
 = \sum_x x \sum_y p(x, y)
 = \sum_x x\\, p_X(x)
 = \mathbb{E}[X].
\end{aligned}
$$

The continuous case is identical with sums replaced by integrals.  QED

**How we use it.**  The Bellman derivation (Section 3.2) conditions on the
state $`S_t = s`$, averages the next transition, and then *re-conditions* the
future return on $`S_{t+1} = s'`$; the law of total expectation is what makes
that double conditioning legitimate.

### 0.2 The Score-Function Lemma (Log-Derivative Trick)

**Theorem (score-function lemma).**  Let $`x \mapsto p_\theta(x)`$ be a
probability density (or mass) function parameterised by $`\theta`$, and let
$`f(x)`$ be any function with finite expectation under $`p_\theta`$.  Then,
provided the interchange of integral and derivative is justified (dominated
convergence),

$$
\large
\nabla_\theta\\, \mathbb{E}_{p_\theta}[f(x)]
= \mathbb{E}_{p_\theta}\\!\left[\\, f(x)\\, \nabla_\theta \log p_\theta(x) \\,\right].
$$

**Proof.**  Differentiate under the integral sign and expand the logarithm:

$$
\large
\begin{aligned}
\nabla_\theta\\, \mathbb{E}_{p_\theta}[f(x)]
&= \nabla_\theta \int f(x)\\, p_\theta(x)\\, dx
 = \int f(x)\\, \nabla_\theta p_\theta(x)\\, dx \\
&= \int f(x)\\, p_\theta(x)\\, \nabla_\theta \log p_\theta(x)\\, dx
 = \mathbb{E}_{p_\theta}\\!\left[\\, f(x)\\, \nabla_\theta \log p_\theta(x) \\,\right].
\end{aligned}
$$

The third equality uses $`\nabla_\theta \log p = (\nabla_\theta p) / p`$
(valid wherever $`p > 0`$; the standard regularity conditions exclude
atoms).  QED

**How we use it.**  It is the entire engine of the policy-gradient theorem
(Section 9): the distribution we sample from is the one we differentiate
through, and the lemma converts "differentiate an expectation over
trajectories" into "multiply the return by the score of the policy".

### 0.3 Banach's Fixed-Point Theorem

**Theorem (Banach fixed-point theorem).**  Let $`X`$ be a complete normed
space (a Banach space), and let $`\mathcal{T} : X \to X`$ be a *contraction*:
there exists $`\gamma \in [0, 1)`$ with

$$
\large
\left\lVert \mathcal{T} x - \mathcal{T} y \right\rVert
\\;\le\\; \gamma\\, \left\lVert x - y \right\rVert
\qquad \forall\\, x, y \in X.
$$

Then $`\mathcal{T}`$ has a **unique fixed point** $`x^{\ast} = \mathcal{T}
x^{\ast}`$; moreover, for every starting point $`x_0 \in X`$, the iterates
$`x_{k+1} = \mathcal{T} x_k`$ converge to it geometrically:

$$
\large
\left\lVert x_k - x^{\ast} \right\rVert
\\;\le\\; \gamma^k\\, \left\lVert x_0 - x^{\ast} \right\rVert.
$$

**Proof.**  *Existence by construction.*  Pick any $`x_0`$ and iterate.
Repeated application of the contraction inequality gives
$`\lVert x_{k+1} - x_k \rVert \le \gamma^k \lVert x_1 - x_0 \rVert`$.  For
$`m > n`$, the triangle inequality telescopes:

$$
\large
\left\lVert x_m - x_n \right\rVert
\\;\le\\; \sum_{j=n}^{m-1} \left\lVert x_{j+1} - x_j \right\rVert
\\;\le\\; \frac{\gamma^n}{1 - \gamma}\\, \left\lVert x_1 - x_0 \right\rVert,
$$

which tends to $`0`$ as $`n \to \infty`$ because $`\gamma^n \to 0`$.  Hence
$`(x_k)`$ is a Cauchy sequence; completeness gives a limit $`x^{\ast}`$.
Continuity of $`\mathcal{T}`$ (contractions are Lipschitz) lets us pass to
the limit in $`x_{k+1} = \mathcal{T} x_k`$:

$$
\large
x^{\ast} = \lim_{k \to \infty} x_{k+1}
= \lim_{k \to \infty} \mathcal{T} x_k
= \mathcal{T} \lim_{k \to \infty} x_k
= \mathcal{T} x^{\ast}.
$$

*Uniqueness.*  If $`x^{\ast}`$ and $`y^{\ast}`$ were both fixed points, then

$$
\large
\left\lVert x^{\ast} - y^{\ast} \right\rVert
= \left\lVert \mathcal{T} x^{\ast} - \mathcal{T} y^{\ast} \right\rVert
\\;\le\\; \gamma\\, \left\lVert x^{\ast} - y^{\ast} \right\rVert.
$$

Since $`\gamma < 1`$, the only possibility is $`\lVert x^{\ast} - y^{\ast}
\rVert = 0`$, i.e. $`x^{\ast} = y^{\ast}`$.  The geometric rate follows by
applying the contraction to the pair $`(x_k, x^{\ast})`$ repeatedly.  QED

**How we use it.**  Section 3.4 shows the Bellman operator is a contraction;
this theorem then delivers existence and uniqueness of the value function in
one line, without constructing it.  Section 12.2 reuses the identical
argument for the soft Bellman operator.

### 0.4 The Maximum-Entropy (KL-Duality) Result

**Lemma.**  Fix a function $`Q(a)`$ over a finite action set and a
temperature $`\alpha > 0`$.  Among all probability distributions $`\pi`$
over actions, the Gibbs distribution

$$
\large
\pi^{\ast}(a) \\;=\\; \frac{\exp\\!\left( Q(a) / \alpha \right)}
{\sum_{a'} \exp\\!\left( Q(a') / \alpha \right)}
$$

is the unique maximizer of the entropy-regularized objective

$$
\large
J(\pi) = \mathbb{E}_{a \sim \pi}\\!\left[\\, Q(a) - \alpha \log \pi(a) \\,\right].
$$

**Proof.**  The objective is concave in $`\pi`$: the term
$`\mathbb{E}_\pi[Q]`$ is linear in $`\pi`$, and the entropy
$`- \sum_a \pi(a) \log \pi(a)`$ is concave, so any stationary point is a
global maximum.  Maximize subject to the simplex constraint
$`\sum_a \pi(a) = 1`$ with a Lagrange multiplier $`\lambda`$:

$$
\large
\mathcal{L}(\pi, \lambda) =
    \sum_a \pi(a)\\, Q(a)
    - \alpha \sum_a \pi(a) \log \pi(a)
    + \lambda \Bigl( 1 - \sum_a \pi(a) \Bigr).
$$

Set the derivative with respect to each $`\pi(a)`$ to zero:

$$
\large
\frac{\partial \mathcal{L}}{\partial \pi(a)}
= Q(a) - \alpha\bigl( \log \pi(a) + 1 \bigr) - \lambda = 0,
\qquad
\pi(a) = \exp\\!\left( \frac{Q(a) - \lambda - \alpha}{\alpha} \right).
$$

The normalization constant absorbs $`\lambda + \alpha`$, which gives the
Gibbs form.  Concavity makes this stationary point the unique maximizer.
QED

**How we use it.**  Section 12.4 derives the optimal soft policy by applying
this lemma with $`Q = Q_{\text{soft}}^{\ast}`$ at every state.

---

# Part 1 — The Problem: Sequential Decisions Under Uncertainty

## 1. What Problem Are We Solving?

A drone flies a delivery route.  At every moment it observes where it is
(`S_t`), picks an action (throttle, yaw, pitch — `A_t`), and the world
responds: the drone has moved (`S_{t+1}`) and it either gained or lost
progress toward the goal **reward** `R_{t+1}`.  The drone wants a *policy*:
a rule that maps observations to actions so that the *total* reward over the
whole flight is as large as possible.

Two features separate RL from ordinary supervised learning:

- **The training signal is a scalar, delayed, and sparse.**  The correct
  action in step $`t`$ is not labelled; the agent discovers it only through
  the *downstream* rewards it causes.  This is the **credit-assignment
  problem**, and it is the entire reason RL has its own mathematics.
- **The agent's behavior changes the data it sees.**  A supervised dataset
  is a fixed file; an RL agent's next observation depends on the action it
  just chose.  The data distribution is *non-stationary and policy-dependent*.

> **What the training loop must therefore provide.**  A component that *acts*
> in an environment and records transitions; a store of past transitions (the
> replay buffer); and a learner that turns those transitions into a better
> policy.  Every practical system decomposes these into three concerns: the
> *environment interaction* owns acting and replay, the *orchestrator* owns
> when to act vs. learn, and the *learner* owns the learning math.  The math
> of the learning math is this document.

---

## 2. The Markov Decision Process, Taught From Scratch

### 2.1 States, Actions, Transitions, Rewards

A **Markov decision process (MDP)** is a tuple
$`(\mathcal{S}, \mathcal{A}, p, r, \gamma)`$:

- $`\mathcal{S}`$ — the set of **states** the environment can be in.
- $`\mathcal{A}`$ — the set of **actions** the agent can take.
- $`p(s' \mid s, a)`$ — the **transition kernel**: the probability that
  taking action $`a`$ in state $`s`$ lands in state $`s'`$.
- $`r(s, a, s')`$ — the **reward** obtained when $`(s, a) \to s'`$.
- $`\gamma \in [0, 1)`$ — the **discount factor**.

A *single step* of the interaction is: the agent observes $`S_t = s`$, picks
$`A_t = a`$, the environment draws $`S_{t+1} = s' \sim p(\cdot \mid s, a)`$
and hands back reward $`R_{t+1} = r(s, a, s')`$.  The sequence

$$
\large
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots
$$

is a **trajectory** (or *rollout*).  The process stops when a terminal state
is reached (e.g. the drone crashed, or the episode ended by a timeout); we
model termination as an absorbing state with zero future reward.

**Rewards are not labels.**  The reward $`r(s,a,s')`$ says how *good it was
to be here and act like this in the moment*; it never says *what the agent
should have done*.  This single fact is what forces every algorithm in this
document.

### 2.2 The Markov Property

The transition kernel $`p(s' \mid s, a)`$ depends on the *current* state and
action only.  The future is conditionally independent of the past given the
present:

$$
\large
p(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots) = p(S_{t+1} \mid S_t, A_t).
$$

This is the **Markov property**.  State $`S_t`$ is said to be *sufficient
statistics of the interaction*: nothing earlier in the trajectory is needed
to predict the next step.  Real environments are rarely exactly Markov
(cruise control that needs last ten seconds of slope), which is why the
*observation* fed to the agent is usually a short history (frame stacking,
velocity buffers).  Everything in this document nevertheless assumes Markov
states $`S_t`$; that assumption is the contract the environment must satisfy.

### 2.3 Policies

A **policy** $`\pi(a \mid s)`$ is a probability distribution over actions for
each state.  Deterministic policies are the special case where one action has
probability 1.  Given $`\pi`$, the MDP becomes a **Markov reward process**
(a Markov chain with rewards): the state sequence alone is Markov with kernel

$$
\large
p_\pi(s' \mid s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)\\, p(s' \mid s, a),
$$

and the expected one-step reward in state $`s`$ is

$$
\large
r_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\, r(s, a, s').
$$

The goal of RL: find the policy $`\pi`$ that maximises expected total
discounted reward (Section 2.4).  We write $`\Pr_\pi`$ and $`\mathbb{E}_\pi`$
for probability and expectation under the trajectories induced by $`\pi`$.

### 2.4 Returns and Discounting

The **return** from time $`t`$ is the discounted sum of future rewards:

$$
\large
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
$$

**Why the discount $`\gamma < 1`$ is not a hack.**  Two reasons, one
mathematical and one economic:

- **Convergence.**  If every reward is bounded, $`|R_t| \le R_{\max}`$, the
  infinite sum is bounded by the geometric series:

$$
\large
\lvert G_t \rvert \\;\le\\; \sum_{k=0}^{\infty} \gamma^k R_{\max}
= \frac{R_{\max}}{1 - \gamma} < \infty.
$$

  **Proof.**  The partial sums satisfy
  $`S_n = R_{\max}(1 - \gamma^{n+1})/(1-\gamma)`$, and
  $`\gamma^{n+1} \to 0`$ as $`n \to \infty`$ because $`0 \le \gamma < 1`$.
  Hence $`S_n \to R_{\max}/(1-\gamma)`$.  Without discounting
  ($`\gamma = 1`$) the infinite-horizon total diverges unless the process
  terminates almost surely; discounting makes *every* infinite-horizon
  problem have a finite objective.
- **Economics / robustness.**  $`\gamma`$ encodes that reward *now* is worth
  more than the same reward later (“a bird in the hand”); it also bounds
  the effect of model error far in the future, which is one reason episodic
  tasks often still use $`\gamma`$ close to 1 (e.g. 0.99) rather than 1.

The return obeys a one-step recursion that is the seed of everything in this
document:

$$
\large
G_t = R_{t+1} + \gamma G_{t+1}.
$$

**Derivation.**  Write the sum twice and factor $`\gamma`$ out of the tail:

$$
\large
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
= R_{t+1} + \gamma\Bigl( R_{t+2} + \gamma R_{t+3} + \cdots \Bigr)
= R_{t+1} + \gamma G_{t+1}.
$$

**In practice.**  The recursion above is used verbatim whenever a bootstrap
target is built (Sections 4 and 6).

### 2.5 Episodes Vs. Continuing Tasks

- **Episodic tasks**: the trajectory ends at a terminal state (game over,
  goal reached, or a timeout).  The per-episode return is a finite sum.
- **Continuing tasks**: no terminal state; returns are always discounted.

In this document we treat every task as an episodic one for bookkeeping:
`done` flags are explicitly propagated through the TD target (the
$(1-d)$ term in Section 4.4).  This matches the standard Gymnasium
convention: the environment owns the details, the learner sees uniform
transition batches.

---

## 3. Value Functions And The Bellman Equations

### 3.1 State-Value and Action-Value

The **state-value function** of a policy $`\pi`$ is the expected return from
state $`s`$:

$$
\large
V^{\pi}(s) = \mathbb{E}_\pi\\!\left[ G_t \\;\middle|\\; S_t = s \right]
= \mathbb{E}_\pi\\!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \\;\middle|\\; S_t = s \right].
$$

The **action-value function** (Q-function) is the expected return from state
$`s`$ after taking action $`a`$ and following $`\pi`$ afterwards:

$$
\large
Q^{\pi}(s, a) = \mathbb{E}_\pi\\!\left[ G_t \\;\middle|\\; S_t = s, A_t = a \right].
$$

The two are related by averaging over the policy's action distribution:

$$
\large
V^{\pi}(s) = \sum_{a} \pi(a \mid s)\\, Q^{\pi}(s, a).
$$

The **optimal** value functions are the pointwise maxima over policies:
$`V^{\ast}(s) = \max_\pi V^\pi(s)`$, $`Q^{\ast}(s,a) = \max_\pi Q^\pi(s,a)`$.

### 3.2 The Bellman Equation for $`V^\pi`$

**Theorem (Bellman equation for a fixed policy).**

$$
\large
V^{\pi}(s) = \sum_{a} \pi(a \mid s)
             \sum_{s'} p(s' \mid s, a)\\,
             \Bigl[\\, r(s, a, s') + \gamma\\, V^{\pi}(s') \Bigr].
$$

**Proof.**  Start from the definition, split off the first reward, and use
the return recursion $`G_t = R_{t+1} + \gamma G_{t+1}`$:

$$
\large
\begin{aligned}
V^{\pi}(s)
&= \mathbb{E}_\pi\\!\left[ G_t \mid S_t = s \right] \\
&= \mathbb{E}_\pi\\!\left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right] \\
&= \sum_{a} \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \Bigl[ r(s, a, s') + \gamma\\, \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \Bigr] \\
&= \sum_{a} \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \Bigl[ r(s, a, s') + \gamma\\, V^{\pi}(s') \Bigr].
\end{aligned}
$$

The third equality uses the law of total expectation and the Markov
property (Section 2.2), which lets the future expectation depend only on
$`S_{t+1} = s'`$.  The fourth line applies the definition of $`V^\pi`$
to the trailing expectation.  The Bellman equation for $`Q^\pi`$ reads

$$
\large
Q^{\pi}(s, a) = \sum_{s'} p(s' \mid s, a)\\,
                \Bigl[ r(s, a, s') + \gamma\\,
                \sum_{a'} \pi(a' \mid s')\\, Q^{\pi}(s', a') \Bigr].
$$

Bellman equations are **consistency conditions**: a function $`V`$ satisfies
the equation *iff* it is the true value of $`\pi`$.  This “if and only
if” is what makes them useful: instead of simulating trajectories to
estimate $`V^\pi`$, we can *solve* the equation.

### 3.3 The Bellman Optimality Equation

The optimal value functions satisfy the **Bellman optimality equations**:

$$
\large
V^{\ast}(s) = \max_{a} \sum_{s'} p(s' \mid s, a)\\,
          \Bigl[ r(s, a, s') + \gamma\\, V^{\ast}(s') \Bigr],
\qquad
Q^{\ast}(s, a) = \sum_{s'} p(s' \mid s, a)\\,
             \Bigl[ r(s, a, s') + \gamma\\, \max_{a'} Q^{\ast}(s', a') \Bigr].
$$

**Theorem (policy improvement).**  Let $`\pi`$ be any policy and let
$`\pi'`$ be the *greedy policy with respect to* $`Q^{\pi}`$:

$$
\large
\pi'(s) \\;\in\\; \arg\max_{a} Q^{\pi}(s, a).
$$

Then $`V^{\pi'}(s) \ge V^{\pi}(s)`$ for every state $`s`$, with a strict
inequality at any state where $`\pi(s)`$ is not a maximizing action.

**Proof.**  By definition of the max,

$$
\large
Q^{\pi}(s, \pi'(s)) = \max_a Q^{\pi}(s, a) \\;\ge\\; Q^{\pi}(s, \pi(s)) = V^{\pi}(s)
$$

for every $`s`$.  Now compare the two policies through their Bellman
operators (Section 3.4).  The operator $`\mathcal{T}^{\pi'}`$ is *monotone*:
if $`f \le g`$ pointwise, then each term
$`\sum_a \pi'(a \mid s) \sum_{s'} p(s' \mid s,a) [\cdots]`$ with its
nonnegative weights preserves the inequality term by term, so
$`\mathcal{T}^{\pi'} f \le \mathcal{T}^{\pi'} g`$.  Starting from
$`V_0 = V^{\pi}`$ and iterating $`V_{k+1} = \mathcal{T}^{\pi'} V_k`$, the
first step satisfies $`V_1 = \mathcal{T}^{\pi'} V^{\pi} \ge V^{\pi} = V_0`$,
because at every state the greedy action gives a one-step lookahead at
least as large as $`\pi`$'s own action.  Monotonicity then gives
$`V_{k+1} \ge V_k`$ for all $`k`$; the contraction property (Section 3.4)
gives $`V_k \to V^{\pi'}`$.  A monotonically increasing sequence converges
from below, so $`V^{\pi'}(s) = \lim_{k \to \infty} V_k(s) \ge V_0(s) =
V^{\pi}(s)`$ for every $`s`$.  QED

**Consequences.**  (i) Repeated greedy improvement produces a nondecreasing
sequence of policies, which must terminate at a policy that is greedy with
respect to its *own* value — exactly the Bellman optimality equation
above — hence at $`\pi^{\ast}`$.  (ii) The optimal policy is *deterministic*:
at the fixed point, a best action exists in every state and the others are
never strictly needed.  This justifies value-based methods that represent
only $`Q`$ and read off $`\pi(s) = \arg\max_a Q(s,a)`$.

> **Sanity check.**  In a grid world where the exit is one step right, the
> Bellman optimality equation says the bonus $`V^{\ast}`$ propagates one cell per
> iteration of the *backward* recursion.  For the exit cell $`V^{\ast}(e) =
> r_e + \gamma \cdot 0`$, for the cell left of it
> $`V^{\ast} = r + \gamma V^{\ast}(e)`$, and so on.  This “three-line” reading of the
> equation is the whole intuition behind dynamic programming AND behind
> Q-learning (Part 2).

### 3.4 Existence, Uniqueness, and the Contraction Property

Define the **Bellman operator** $`\mathcal{T}^\pi`$ on the space of bounded
real-valued functions over states, componentwise:

$$
\large
\left(\mathcal{T}^\pi V\right)(s)
= \sum_{a} \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
  \Bigl[ r(s, a, s') + \gamma\\, V(s') \Bigr].
$$

**Theorem (contraction).**  $`\mathcal{T}^\pi`$ is a $`\gamma`$-contraction
in the sup-norm $`\lVert V \rVert_\infty = \max_s |V(s)|`$:

$$
\large
\left\lVert \mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2 \right\rVert_\infty
\\;\le\\; \gamma\\, \left\lVert V_1 - V_2 \right\rVert_\infty.
$$

**Proof.**  For any state $`s`$,

$$
\large
\begin{aligned}
\Bigl|\left(\mathcal{T}^\pi V_1\right)(s) - \left(\mathcal{T}^\pi V_2\right)(s)\Bigr|
&= \Bigl|\\,\sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \gamma\\, \bigl(V_1(s') - V_2(s')\bigr)\Bigr| \\
&\le \gamma \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \left\lvert V_1(s') - V_2(s') \right\rvert \\
&\le \gamma \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \left\lVert V_1 - V_2 \right\rVert_\infty
= \gamma\\, \left\lVert V_1 - V_2 \right\rVert_\infty.
\end{aligned}
$$

The first inequality is the triangle inequality; the second replaces each
absolute value by the global sup-norm; the final equality uses that the
double sum of probabilities equals 1.

**Consequences (Banach fixed-point theorem, proved in Section 0.3).**
Because $`\gamma < 1`$, $`\mathcal{T}^\pi`$ is a contraction on a complete
metric space, so by Banach's theorem it has a **unique fixed point**, and
iteration from any starting $`V_0`$ converges to it at a geometric rate:

$$
\large
V^\pi = \lim_{k \to \infty} \left(\mathcal{T}^\pi\right)^k V_0,
\qquad
\left\lVert V_k - V^\pi \right\rVert_\infty \le \gamma^k
\left\lVert V_0 - V^\pi \right\rVert_\infty.
$$

This is *policy evaluation* (dynamic programming) and it is guaranteed to
work — when we know $`p`$ and $`r`$.  RL exists precisely because we
usually do not.  Part 2 replaces the sums over $`p`$ with *samples*.

The same contraction argument, on the space of *action-value* functions
and with the *max* operator, will reappear as the engine of Q-learning
convergence (Section 5.2).

---

# Part 2 — Learning From Experience: Temporal-Difference Methods

## 4. Monte Carlo, Bootstrapping, And The TD Error

### 4.1 What We Are Allowed To Touch

In the model-free setting we do **not** know $`p(s' \mid s, a)`$ or
$`r(s, a, s')`$.  What we *can* do is interact with the environment and
observe outcomes: states, actions, rewards.  Every algorithm in Parts 2–5
has the same skeleton:

1. *Generate/collect experience* under some behaviour policy.
2. *Use the experience to move a learned estimate* (a table, later a neural
   network) toward a target derived from the Bellman equations.

The only real choice is *what target*, and *how much of it is bootstrapped*.

### 4.2 Monte Carlo (MC): The Target Is A Complete Trajectory Return

The simplest idea: run an episode to completion, accumulate the return
$`G_t`$ from each visited state, and move the estimate of $`V(s_t)`$ toward
$`G_t`$:

$$
\large
V(s_t) \\;\leftarrow\\; V(s_t) + \alpha\\, \bigl[\\, G_t - V(s_t) \\,\bigr],
\qquad \alpha \in (0, 1].
$$

This is **Monte Carlo policy evaluation**.  The update is unbiased:
$`\mathbb{E}[G_t \mid S_t = s] = V^\pi(s)`$ by definition, so in expectation
the update moves $`V(s)`$ toward its true value.

**Cost of unbiasedness.**  MC needs a *completed* episode (until
termination), and the return $`G_t`$ has high variance: it is a sum of many
random rewards, so any single sampled episode can be a poor estimate of the
mean.  Also, in continuing (non-episodic) tasks the return is infinite
without discounting, and even with discounting one must truncate.

### 4.3 Temporal Difference (TD): The Target Bootstrap

The TD idea: replace the *full* return $`G_t`$ with a *one-step*
bootstrapped target that reuses the current estimate of the future:

$$
\large
V(s_t) \\;\leftarrow\\; V(s_t) + \alpha\\,
\bigl[\\, R_{t+1} + \gamma\\, V(s_{t+1}) - V(s_t) \\,\bigr].
$$

The quantity in brackets,

$$
\large
\delta_t \\;=\\; R_{t+1} + \gamma\\, V(s_{t+1}) - V(s_t),
$$

is the **TD error** at time $`t`$.  The target
$`R_{t+1} + \gamma V(s_{t+1})`$ is called the **TD target**.

**Why the name.**  $`R_{t+1} + \gamma V(s_{t+1})`$ is a *one-step sample*
of the Bellman expectation in Section 3.2: we replace the inner expectation
over $`s'`$ and $`a'`$ with a single observed transition, and the *unknown*
$`V(s')`$ with our current estimate.  TD therefore *bootstraps* (it uses an
estimate in its own target) — unlike MC, which uses only observed rewards.

**Theorem (TD target is a lower-variance, biased estimator of $`V^\pi`$).**
  Let $`\delta_t`$ be the TD error defined above.  If $`V`$ is the *true*
  value function $`V^\pi`$, then $`\mathbb{E}[\delta_t] = 0`$.

**Proof.**  Condition on $`S_t = s`$, and let $`A_t \sim \pi(\cdot \mid s)`$,
$`S_{t+1} \sim p(\cdot \mid s, A_t)`$, $`R_{t+1} = r(S_t, A_t, S_{t+1})`$:

$$
\large
\begin{aligned}
\mathbb{E}\left[\delta_t \mid S_t = s\right]
&= \mathbb{E}\left[R_{t+1}\right] + \gamma\\,
   \mathbb{E}\left[V^\pi(S_{t+1})\right] - V^\pi(s) \\
&= \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\\,
   \Bigl[ r(s,a,s') + \gamma\\, V^\pi(s') \Bigr] - V^\pi(s).
\end{aligned}
$$

The double sum on the right is exactly $`\left(\mathcal{T}^\pi
V^\pi\right)(s) = V^\pi(s)`$ by the Bellman equation (Section 3.2), so the
difference is $`0`$.  Hence the TD update is *unbiased when the estimate is
correct* (a property called consistency at the fixed point), and its
variance is much lower than MC because only *one* random transition (and one
bootstrapped estimate) enter the target.  The classic trade: MC is
unbiased but high-variance; TD is biased-until-converged but low-variance.
In practice, TD wins on non-stationary / long-horizon problems, which is why
deep RL is overwhelmingly TD-based.

### 4.4 Why Ordinary TD Needs a `done` Flag

The TD target uses $`V(s_{t+1})`$ as the value of the *next* state.  After
the terminal state $`s_T`$ (the episode ended) there is no next state and
no future reward; the correct target for the last transition is just
$`R_T`$.  In one formula, with the *done* indicator $`d_t = 1`$ iff
$`S_{t+1}`$ is terminal:

$$
\large
\text{TD target} \\;=\\; R_{t+1} + \gamma\\, (1 - d_t)\\, V(s_{t+1}).
$$

**Proof.**  By definition of a terminal state, $`V(\text{terminal}) = 0`$:
no future reward accrues from it, and the process stops, so the expected
discounted continuation is $`0`$.  The indicator is $`d_t = 1`$ exactly
when $`S_{t+1}`$ is terminal and $`0`$ otherwise.  Hence, in every case,

$$
\large
(1 - d_t)\\, V(s_{t+1})
= \begin{cases}
    V(s_{t+1}), & S_{t+1} \text{ non-terminal},\\
    0,          & S_{t+1} \text{ terminal},
  \end{cases}
$$

which is precisely the value of the continuation used by the TD target:
when the episode ends there is no next state to bootstrap from, so the
target must reduce to the observed reward $`R_{t+1}`$ alone, exactly as
the formula $`R_{t+1} + \gamma (1 - d_t) V(s_{t+1})`$ does.  QED

This masking step is the single most common correctness bug in RL
implementations; every practical target builder multiplies the bootstrap
term by `1 - done` *before* applying the discount.

### 4.5 The General n-Step View (and why 1-step is the deep-RL default)

Between MC and 1-step TD lies a family: use $`n`$ observed rewards and
bootstrap beyond that:

$$
\large
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}
            + \gamma^n V(s_{t+n}).
$$

$`n = 1`$ is TD; $`n = \infty`$ is MC (until termination).  Larger $`n`$
is less biased but more variable.  Deep RL defaults to $`n = 1`$ with a
target network (Part 3) for stability, and occasionally uses $`n = 2`$-3
or a sampled geometric mixture (e.g. $`\lambda`$-returns; see
Section 10.4), because the target network already removes most of the
instability that multi-step targets would reintroduce.

In practice several libraries expose $`n`$-step targets as a drop-in
replacement for the one-step bootstrap; the implementation simply expands
the reward sequence by $`n`$ steps and bootstraps on $`Q`$ at time $`t+n`$.

---

## 5. Q-Learning: Off-Policy Control By A Single-Step Bootstrap

### 5.1 The Update Rule

Q-learning estimates the *optimal* action-value function directly, using
the *max* operator from the Bellman optimality equation (Section 3.3):

$$
\large
Q(s_t, a_t) \\;\leftarrow\\; Q(s_t, a_t) + \alpha\\,
\Bigl[\\, R_{t+1} + \gamma\\, (1 - d_t)\\, \max_{a'} Q(s_{t+1}, a')
\\;-\\; Q(s_t, a_t) \\,\Bigr].
$$

The behaviour policy (how actions are chosen in the environment) and the
target policy (the greedy $`\max`$) are **different** — that is what
*off-policy* means.  We can act with, say, an *exploratory*
$`\varepsilon`$-greedy policy, yet learn the value of the *greedy* policy,
because the update uses $`\max_{a'} Q(s_{t+1}, a')`$ and never the
behaviour policy's next-action probabilities.

### 5.2 Why It Converges: Stochastic Approximation

**Setup.**  Write the update with explicit step size and index:

$$
\large
Q_{t+1}(s, a)
= Q_t(s, a) + \alpha_t(s, a)\\,
\Bigl[\\, R_{t+1} + \gamma\\, \max_{a'} Q_t(s', a') - Q_t(s, a) \\,\Bigr],
$$

where $`(s, a, s')`$ is the transition sampled at step $`t`$ under the
behaviour policy, and $`\alpha_t(s, a)`$ is a step size that is
$`0`$ whenever the pair $`(s, a)`$ is not updated at step $`t`$.

**Step 1: the update is a noisy fixed-point iteration.**  Define the
*optimality operator* $`\mathcal{T}^{\ast}`$ on action-value functions by

$$
\large
\bigl(\mathcal{T}^{\ast} Q\bigr)(s, a)
= \sum_{s'} p(s' \mid s, a)\\,
  \Bigl[\\, r(s, a, s') + \gamma\\, \max_{a'} Q(s', a') \\,\Bigr].
$$

This is the Bellman optimality equation's right-hand side: its fixed point
is $`Q^{\ast}`$ (Section 3.3).  The sample in brackets is an *unbiased*
estimate of $`(\mathcal{T}^{\ast} Q_t)(s, a)`$, because the transition is
drawn from $`p(\cdot \mid s, a)`$ (the behaviour policy only decides *which*
pair is visited; the conditional distribution of $`s'`$ given $`(s,a)`$ is
the fixed environment kernel).  Write the noise explicitly:

$$
\large
R_{t+1} + \gamma\\, \max_{a'} Q_t(s', a') - Q_t(s, a)
= \bigl(\mathcal{T}^{\ast} Q_t\bigr)(s, a) - Q_t(s, a) + w_t,
$$

with $`w_t`$ a zero-mean error conditional on the entire history up to
$`t`$.  The update is therefore a **stochastic approximation** of the
deterministic iteration $`Q \leftarrow (1 - \alpha) Q + \alpha
\mathcal{T}^{\ast} Q`$.

**Step 2: the deterministic iteration converges.**  The operator
$`\mathcal{T}^{\ast}`$ is a $`\gamma`$-contraction in the supremum norm.
The proof is the one from Section 3.4 with the key inequality

$$
\large
\Bigl\lvert \max_a f(a) - \max_a g(a) \Bigr\rvert
\\;\le\\; \max_a \bigl\lvert f(a) - g(a) \bigr\rvert
\\;\le\\; \lVert f - g \rVert_\infty,
$$

which holds because both maxima lie between the two functions' pointwise
range.  Hence, exactly as in Section 3.4,

$$
\large
\bigl\lVert \mathcal{T}^{\ast} Q_1 - \mathcal{T}^{\ast} Q_2 \bigr\rVert_\infty
\\;\le\\; \gamma\\, \bigl\lVert Q_1 - Q_2 \bigr\rVert_\infty,
$$

so by Banach's theorem (Section 0.3) the deterministic iteration converges
to $`Q^{\ast}`$ from any start, and the noise-free step
$`Q \leftarrow (1-\alpha)Q + \alpha \mathcal{T}^{\ast} Q`$ does too (it is
an affine contraction with the same modulus).

**Step 3: the noise averages out (Robbins–Monro).**  The random
algorithm perturbs the contracting iteration by a zero-mean noise at every
step.  The classical conditions that make such perturbations vanish are the
**Robbins–Monro step-size conditions**:

$$
\large
\sum_{t=0}^{\infty} \alpha_t(s, a) = \infty,
\qquad
\sum_{t=0}^{\infty} \alpha_t(s, a)^2 < \infty
\qquad \text{(for every } s, a \text{ visited infinitely often).}
$$

The first condition guarantees the algorithm does not stop improving; the
second guarantees the total noise power is finite, so the cumulative
perturbation $`\sum_t \alpha_t w_t`$ converges a.s. to a finite value
(Kolmogorov's criterion for martingale differences applied to $`\alpha_t
w_t`$).  Combining the geometric attraction of Step 2 with the vanishing
perturbation yields the standard convergence theorem (Jaakkola, Jordan, and
Singh 1994; Watkins and Dayan 1992; see Appendix C).

**Theorem (Q-learning convergence).**  For a finite MDP, if
$`\sum_t \alpha_t(s,a) = \infty`$ and $`\sum_t \alpha_t(s,a)^2 < \infty`$
a.s. for every state–action pair visited infinitely often (e.g. with
$`\varepsilon`$-greedy exploration, where every pair is visited i.o.), then
$`Q_t \to Q^{\ast}`$ almost surely.

**Why exploration matters here.**  The step-size conditions alone are not
enough: the pair $`(s, a)`$ must actually be *visited* infinitely often for
its estimate to keep being updated.  This is exactly what
$`\varepsilon`$-greedy (or any policy with $`\varepsilon > 0`$) guarantees,
and it is the mathematical reason RL separates *behaviour* policy from
*target* policy: the behaviour policy must cover all actions, while the
target policy is greedy.

**Why the max is not cheating (the exploration question).**  If we always
act greedily with respect to a not-yet-accurate $`Q`$, we may lock in a
suboptimal action forever: *exploitation without exploration*.  The standard
cure is $`\varepsilon`$-greedy behaviour: with probability
$`1 - \varepsilon`$ act greedily, with probability $`\varepsilon`$ act
uniformly at random.  The *learned* object $`Q`$ is still the greedy value;
the *data-generating* policy is $`\varepsilon`$-greedy.  Deep RL anneals
$`\varepsilon`$ from ~1.0 to a small floor (0.05) so that early exploration
is broad and later behaviour concentrates on the learned policy.

The key obstruction in the deep case (Part 3) is exactly that a neural
network breaks the *contraction* (its update is no longer a contraction),
which is why deep Q-learning needs replay and target networks to recover
*approximate* stability — the stochastic-approximation guarantee above does
not survive function approximation unchanged.

**Tabular Q-learning is not the destination.**  Real tasks have state spaces
far too large for tables.  Part 3 replaces $`Q(s,a)`$ with
$`Q_\theta(s,a)`$, a neural network with parameters $`\theta`$, and turns
the scalar update above into a regression loss.

---

# Part 3 — Deep Q-Learning

## 6. Function Approximation, Replay, And Target Networks

### 6.1 From Table To Network

A table $`Q(s,a)`$ over all state–action pairs is impossible when states are
images or high-dimensional vectors.  We replace the table with a parametric
function $`Q_\theta(s, a)`$ — a neural network with weights $`\theta`$
(a **Q-network**).  The update in Section 5.1 becomes a *regression step*:
move $`Q_\theta`$ toward the TD target on each sampled transition.

A Q-network is a small composite module:

- a **backbone** — an encoder (for image observations) or a plain MLP
  (for vector observations); and
- a **Q head** — a linear layer `backbone_dim -> n_actions`, or a
  **dueling** head `(value + advantage streams)` (Section 8).

Its `forward(s)` returns the vector of action values
$`Q_\theta(s, \cdot) \in \mathbb{R}^{|\mathcal{A}|}`$, so
`argmax_a Q_\theta(s, a)` is one `.argmax()` call.

### 6.2 The DQN Loss

Given a mini-batch of transitions $`\{(s_i, a_i, r_i, s'_i, d_i)\}`$, DQN
(Mnih et al. 2015, [4]) minimizes the mean squared TD error:

$$
\large
\mathcal{L}(\theta)
= \frac{1}{N} \sum_{i=1}^{N}
  \Bigl[\\,
    \underbrace{r_i + \gamma (1 - d_i)\\, \max_{a'} Q_{\theta^-}(s'_i, a')}_{\text{TD target } y_i}
    \\;-\\; Q_\theta(s_i, a_i)
  \Bigr]^2 .
$$

The inner regression loss is the **Bellman residual** applied to samples;
the sum over a mini-batch is the function-approximation analogue of the
scalar TD error of Section 4.3.

### 6.3 Why Replay? (The Three Deadly Sins Of Online RL)

Naively, we could update $`\theta`$ on each transition as it arrives, then
discard it.  Three pathologies make this fail in practice; experience replay
fixes all three.

**1. Correlated data.**  Consecutive transitions are strongly correlated
(the drone's state changes by a small amount each step).  SGD on correlated
data over-fits the recent region and forgets the past.  Replay **decorrelates**
the batch: each mini-batch is sampled uniformly from a buffer of the last
$`N`$ transitions, so no two batch elements are temporally adjacent.

**2. Non-stationarity of the target.**  The TD target
$`r + \gamma \max Q_\theta(s', \cdot)`$ depends on the *same* weights $`\theta`$
being updated.  Learning chases a moving target, which is a recipe for
oscillation or divergence.  This is the core problem for which the **target
network** below is the fix: the target uses frozen weights $`\theta^-`$.

**3. Policy–data feedback.**  The policy improves, so the distribution of
visited states changes.  Replay decouples *learning* from *acting*: the
learner sees a mixture of old and new experience, which keeps early
discoveries from being forgotten (reduces *catastrophic interference*).

In practice the buffer is a fixed-capacity circular store of transition
tuples $`(s, a, r, s', d)`$, sampled uniformly at random
(**experience replay**).  Following **prioritized experience replay**
(Schaul et al. 2015, [5]), a common refinement samples with probability
proportional to $`|\delta_i|^\alpha`$, the magnitude of the TD error, and
corrects the induced bias with importance-sampling weights
$`w_i \propto (N \cdot P(i))^{-\beta}`$.

### 6.4 Target Networks: The Stationarity Fix

Define a second network $`Q_{\theta^-}`$ — the **target network** — whose
weights $`\theta^-`$ are *not* updated by gradient descent.  The loss uses
$`Q_{\theta^-}`$ only *inside the target* $`y`$; gradients flow only into
$`\theta`$:

$$
\large
\nabla_\theta \mathcal{L}(\theta)
= -\frac{2}{N} \sum_i
  \Bigl[ y_i - Q_\theta(s_i, a_i) \Bigr]\\,
  \nabla_\theta Q_\theta(s_i, a_i),
\qquad
y_i = r_i + \gamma (1 - d_i)\\, \max_{a'} Q_{\theta^-}(s'_i, a').
$$

Because $`\theta^-`$ moves slowly (either hard-copied every $`C`$ steps or
Polyak-averaged every step, Section 6.5), the target $`y_i`$ is
*approximately stationary* over short horizons — the moving-target problem
of Section 6.3.2 is mitigated.  The target network is thus the deep analogue
of the *contraction* property that made tabular Q-learning converge
(Section 5.2): a frozen target restores the Bellman operator as a fixed
point the learner can chase.

> **Why deepcopy, not a second optimizer.**  The target net is a structural
> *copy* of the online net, not a separately trained model: create it once as
> `copy.deepcopy(online)` and freeze it (`requires_grad_(False)`).  Its
> parameters are typically stored alongside the online ones (e.g. under a
> `target.*` prefix), so one state dict carries both.

### 6.5 Two Target-Update Schedules

- **Hard update.**  Every $`C`$ steps, copy $`\theta \to \theta^-`$.
  Simple, and what the original DQN paper used ($`C = 10^4`$ steps).
- **Soft (Polyak) update.**  Every step,
  $`\theta^- \\;\leftarrow\\; \tau\\, \theta + (1 - \tau)\\, \theta^-`$
  with small $`\tau`$ (e.g. $`0.005`$).  Smooth, and standard in modern
  actor-critic methods (SAC, TD3).  Because it is a convex combination,
  and $`\theta^-`$ and $`\theta`$ share the same parameter space, the
  moving average stays a valid Q-function if the online one is.

Both schedules appear in practice: hard updates pick a frequency $`C`$;
soft updates run every step.

### 6.6 Exploration: The $`\varepsilon`$-Greedy Schedule

The behaviour policy acts greedily with probability
$`1 - \varepsilon_t`$ and uniformly at random with probability
$`\varepsilon_t`$.  DQN anneals $`\varepsilon`$ linearly from
$`\varepsilon_{\text{start}}`$ (usually 1.0) to $`\varepsilon_{\text{end}}`$
(usually 0.05) over a fixed number of steps $`T_{\text{decay}}`$:

$$
\large
\varepsilon_t
= \varepsilon_{\text{end}} + \bigl(\varepsilon_{\text{start}} - \varepsilon_{\text{end}}\bigr)\\,
  \max\\!\Bigl(0,\\; 1 - \tfrac{t}{T_{\text{decay}}}\Bigr).
$$

The schedule depends on the global step count; a resumed run continues the
annealing instead of restarting it.

---

## 7. Double DQN: Fixing Overestimation

### 7.1 The Problem: Maximization Bias

The DQN target uses $`\max_{a'} Q_{\theta^-}(s', a')`$.  The max of noisy
estimates is a biased, *upward*-biased, estimator of the max of the true
values:

**Lemma (maximization bias).**  Let $`X_1, \dots, X_k`$ be unbiased
estimators of $`\mu_1, \dots, \mu_k`$.  Then

$$
\large
\mathbb{E}\\!\left[\max_i X_i\right] \\;\ge\\; \max_i \mu_i,
$$

with strict inequality whenever the $`X_i`$ are not perfectly correlated and
have positive variance.

**Proof.**  For any realization, $`\max_i X_i \ge X_j`$ for every $`j`$.
In particular $`\max_i X_i \ge X_{j^{\ast}}`$ where $`j^{\ast} = \arg\max_i \mu_i`$.
Take expectations: $`\mathbb{E}[\max_i X_i] \ge \mathbb{E}[X_{j^{\ast}}] =
\mu_{j^{\ast}} = \max_i \mu_i`$.  Equality holds only if the realized max is
always achieved by $`j^{\ast}`$ — false with positive variance, since then some
other $`X_j`$ occasionally exceeds it.  Hence the strict upward bias.

In Q-learning the noise is the estimation error of $`Q_{\theta^-}`$
(which is real: the network is imperfect, especially early in training), so
the greedy target systematically overestimates true action values.  In
practice this causes over-optimistic value estimates and, through the
policy, suboptimal action selection.

### 7.2 The Fix: Decouple Selection From Evaluation

Double Q-learning (Hasselt 2010, [6]) keeps *two* estimates and uses one to
*select*, the other to *evaluate*:

$$
\large
y_i^{\text{Double}}
= r_i + \gamma (1 - d_i)\\,
  Q_{\theta^-}\\!\Bigl(s'_i,\\; \arg\max_{a'} Q_\theta(s'_i, a')\Bigr).
$$

The **online** network selects the best action at $`s'`$:
$`a^{\ast} = \arg\max_{a'} Q_\theta(s', a')`$.  The **target** network then
*evaluates* that chosen action: $`Q_{\theta^-}(s', a^{\ast})`$.  Selection and
evaluation use different weights, so the upward max bias of Section 7.1 is
largely cancelled (the estimator whose max is taken — $`Q_{\theta^-}`$ — is
evaluated at a point chosen by the other network, breaking the
"max of noisy" structure).

In code, the two networks are conventionally distinguishable: the target
lives behind a `double_q` flag that selects the index with the *online*
network and evaluates it with the *target* network — ensuring the two
cannot be accidentally swapped.

---

## 8. Dueling DQN: The Advantage Decomposition

### 8.1 Advantage Functions

Define the **advantage** of action $`a`$ in state $`s`$
(Section 3.1 relation $`V(s) = \sum_a \pi(a \mid s) Q(s,a)`$):

$$
\large
A(s, a) \\;=\\; Q(s, a) - V(s).
$$

$`A(s,a)`$ measures how much *better or worse than average* action $`a`$
is in state $`s`$.  Crucially, for the greedy (or, more generally, any)
policy, the expectation of $`A`$ over $`a \sim \pi`$ is zero:

$$
\large
\sum_a \pi(a\mid s)\\, A(s,a)
= \sum_a \pi(a\mid s)\\, Q(s,a) - V(s)\\,
 \underbrace{\sum_a \pi(a\mid s)}_{1}
= V(s) - V(s) = 0.
$$

### 8.2 The Dueling Architecture

The dueling network (Wang et al. 2016, [7]) parameterizes the Q-function as
a sum of a state-value stream and an advantage stream:

$$
\large
Q_\theta(s, a) \\;=\\; V_\eta(s) \\;+\\; A_\psi(s, a).
$$

But this decomposition is **not identifiable**: adding a constant $`c(s)`$
to all advantages and subtracting it from $`V`$ leaves $`Q`$ unchanged.
The model can represent the same $`Q`$ with infinitely many parameter
pairs.  The standard fix forces the advantages to sum to zero by subtracting
their *mean*:

$$
\large
Q_\theta(s, a)
= V_\eta(s) + A_\psi(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A_\psi(s, a').
$$

**Claim.**  The mean-subtracted form makes the decomposition identifiable
(no redundancy), retains full expressive power, and satisfies the identity
$`\max_a Q_\theta(s,a) = V_\eta(s) + \max_a \tilde{A}_\psi(s,a)`$ with
$`\tilde{A}_\psi = A_\psi - \mathrm{mean}(A_\psi)`$.

**Proof.**  Write $`\bar{A}(s) = \frac{1}{|\mathcal{A}|} \sum_{a'}
A_\psi(s,a')`$ and define the effective advantage
$`\tilde{A}_\psi(s,a) = A_\psi(s,a) - \bar{A}(s)`$.  By construction
$`\sum_a \tilde{A}_\psi(s,a) = \sum_a A_\psi(s,a) - |\mathcal{A}|
\bar{A}(s) = 0`$, matching the zero-mean property of true advantages
(Section 8.1) and pinning down $`V_\eta`$ uniquely: averaging $`Q_\theta`$
over $`a`$ gives
$`\frac{1}{|\mathcal{A}|} \sum_a Q_\theta(s,a) = V_\eta(s)`$.

For the max identity, add and subtract the mean inside the max:

$$
\large
\begin{aligned}
\max_a Q_\theta(s, a)
&= V_\eta(s) + \max_a \bigl[ A_\psi(s,a) - \bar{A}(s) \bigr] \\
&= V_\eta(s) + \max_a \tilde{A}_\psi(s,a).
\end{aligned}
$$

Finally, full expressiveness: given any true $`Q^{\ast}`$, set
$`V_\eta(s) = \max_a Q^{\ast}(s,a)`$ and
$`A_\psi(s,a) = Q^{\ast}(s,a) - \max_{a'} Q^{\ast}(s,a')`$.  Then
$`A_\psi \le 0`$ and $`\max_a A_\psi(s,a) = 0`$, so the mean-subtracted sum
reproduces $`Q^{\ast}`$ exactly.  QED

**Why it helps sample efficiency.**  States where the *relative* value of
actions is unimportant (long corridors in a maze, straight road sections)
need not waste capacity modelling per-action differences: the value stream
$`V_\eta`$ can learn the "how good is this state" signal on its own, and
the advantage stream only refines what differs across actions.  Ablations
in [7] show improved stability and faster learning, especially in
action-sparse regimes.

In code the mean-subtraction term is applied inside the head so the
network always outputs well-defined action values.

---

# Part 4 — Policy Gradients And PPO

## 9. The Policy Gradient Theorem

### 9.1 From Values To Policies

Value-based methods (Parts 2–3) learn $`Q`$ and derive the policy as
$`\arg\max_a Q(s,a)`$.  **Policy-gradient methods** instead parameterize the
policy directly: $`\pi_\theta(a \mid s)`$, a differentiable function of
$`\theta`$ (a softmax head, or a Gaussian over a continuous action space).
The objective is the expected discounted return under $`\pi_\theta`$:

$$
\large
J(\theta) \\;=\\; \mathbb{E}_{\pi_\theta}\\!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \right]
\\;=\\; \mathbb{E}_{\pi_\theta}\\!\left[ G_t \right].
$$

We want $`\nabla_\theta J(\theta)`$.  The apparent obstacle: the
expectation is over trajectories sampled from $`\pi_\theta`$, and the
distribution itself depends on $`\theta`$ — the gradient must account for
the *path dependence* of the sampling distribution, not just the rewards.

### 9.2 The Score-Function Trick

The engine of the whole section is a standard probabilistic identity.

**Lemma (score function).**  For a parameterized density $`p_\theta(x)`$
(discrete or continuous) which is positive on its support,

$$
\large
\mathbb{E}_{p_\theta}\\!\left[ \nabla_\theta \log p_\theta(x) \right] = 0,
\qquad
\nabla_\theta \mathbb{E}_{p_\theta}\\!\left[ f(x) \right]
= \mathbb{E}_{p_\theta}\\!\left[ f(x)\\, \nabla_\theta \log p_\theta(x) \right]
$$

for any function $`f`$ (under regularity conditions allowing the exchange
of gradient and integral).

**Proof.**  Start from $`\int p_\theta(x)\\, dx = 1`$ and differentiate:

$$
\large
0 = \nabla_\theta \int p_\theta(x)\\, dx
= \int \nabla_\theta p_\theta(x)\\, dx
= \int p_\theta(x)\\, \nabla_\theta \log p_\theta(x)\\, dx,
$$

where the last step uses $`\nabla_\theta \log p_\theta = \nabla_\theta
p_\theta / p_\theta`$.  For the second identity, differentiate
$`\mathbb{E}[f] = \int f(x)\\, p_\theta(x)\\, dx`$:

$$
\large
\nabla_\theta \mathbb{E}_{p_\theta}[f]
= \int f(x)\\, \nabla_\theta p_\theta(x)\\, dx
= \int f(x)\\, p_\theta(x)\\, \nabla_\theta \log p_\theta(x)\\, dx
= \mathbb{E}_{p_\theta}\\!\left[ f(x)\\, \nabla_\theta \log p_\theta(x) \right].
$$

The quantity $`\nabla_\theta \log p_\theta(x)`$ is the **score** of $`p_\theta`$.

### 9.3 The Policy Gradient Theorem (trajectory form)

**Theorem (policy gradient, trajectory form; Sutton et al. 1999, [8]).**
  For a stochastic policy $`\pi_\theta`$, the gradient of $`J`$ is

$$
\large
\nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta}\\!\left[ G_t\\, \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right],
$$

where $`G_t`$ is the return from time $`t`$.

**Proof.**  A trajectory $`\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)`$ has
probability

$$
\large
p_\theta(\tau) = p(S_0) \prod_{t \ge 0} \pi_\theta(A_t \mid S_t)\\, p(S_{t+1} \mid S_t, A_t).
$$

Only the policy factors depend on $`\theta`$; the transition kernel
$`p(S_{t+1} \mid S_t, A_t)`$ and initial state distribution are environment-
given and independent of $`\theta`$.  Taking the score,

$$
\large
\begin{aligned}
\nabla_\theta \log p_\theta(\tau)
&= \nabla_\theta \Bigl[ \log p(S_0) + \sum_{t \ge 0} \log \pi_\theta(A_t \mid S_t) + \sum_{t \ge 0} \log p(S_{t+1} \mid S_t, A_t) \Bigr] \\
&= \sum_{t \ge 0} \nabla_\theta \log \pi_\theta(A_t \mid S_t).
\end{aligned}
$$

Now apply the score-function lemma with $`x = \tau`$, $`f = G`$:

$$
\large
\nabla_\theta J(\theta)
= \mathbb{E}_\tau\\!\left[ G(\\tau)\\, \nabla_\theta \log p_\theta(\tau) \right]
= \mathbb{E}_\tau\\!\left[ G(\\tau) \sum_{t \ge 0} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right].
$$

Summing over $`t`$ inside the return and using linearity of expectation
gives the claimed (equivalent) form.  The theorem is the *why* behind the
classic REINFORCE update:

$$
\large
\theta \\;\leftarrow\\; \theta + \alpha\\, G_t\\, \nabla_\theta \log \pi_\theta(A_t \mid S_t).
$$

### 9.4 Reducing Variance: Baselines and the Advantage

The raw REINFORCE estimate
$`G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)`$ is unbiased but has very
high variance: the return $`G_t`$ varies wildly across trajectories.

**Theorem (baseline invariance).**  Subtracting a function $`b(s)`$ that
depends only on the *state* does not change the gradient in expectation:

$$
\large
\mathbb{E}_{\pi_\theta}\\!\left[ \bigl(G_t - b(S_t)\bigr)\\,
\nabla_\theta \log \pi_\theta(A_t \mid S_t) \right)
= \mathbb{E}_{\pi_\theta}\\!\left[ G_t\\, \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right].
$$

**Proof.**  It suffices to show the added term has zero mean.  Condition on
$`S_t = s`$ and use the score-function lemma (Section 9.2) on the conditional
distribution $`\pi_\theta(\cdot \mid s)`$:

$$
\large
\begin{aligned}
\mathbb{E}_{\pi_\theta}\\!\left[ b(S_t)\\, \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right]
&= \mathbb{E}_{S_t}\\!\left[ b(s)\\,
   \mathbb{E}_{A_t \sim \pi_\theta(\cdot \mid s)}\\!\left[
     \nabla_\theta \log \pi_\theta(A_t \mid s) \right] \right] \\
&= \mathbb{E}_{S_t}\\!\left[ b(s)\\, 0 \right] = 0.
\end{aligned}
$$

The inner expectation is zero by the score-function lemma applied to
$`\pi_\theta(\cdot \mid s)`$.

The **best** state-dependent baseline is the state-value $`V^{\pi_\theta}(s)`$,
because it is the conditional mean of the return.  With it, the policy
gradient becomes

$$
\large
\nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta}\\!\left[
  \bigl( G_t - V^{\pi_\theta}(S_t) \bigr)\\,
  \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right],
$$

and the quantity $`G_t - V^{\pi_\theta}(S_t)`$ is exactly the return-based
advantage — a low-variance, zero-mean-per-state signal, the natural successor
of the Section 8.1 advantage.  This is the **actor-critic** idea: the *actor*
is $`\pi_\theta`$, the *critic* is the learned value function used as a
baseline.

---

## 10. Actor-Critic And Generalized Advantage Estimation

### 10.1 The One-Step Actor-Critic

Replacing the full return $`G_t`$ in the policy gradient by the one-step TD
target is the **one-step actor-critic**:

$$
\large
\nabla_\theta J(\theta)
\approx \mathbb{E}\\!\left[
  \Bigl( R_{t+1} + \gamma\\, V_\phi(s_{t+1}) - V_\phi(s_t) \Bigr)\\,
  \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right],
$$

where the critic $`V_\phi`$ (weights $`\phi`$, trained by TD, Section 4.3)
is both the baseline and the bootstrap.  The bracket is the *TD error* —
advantage with bootstrapping.  This single formula is the ancestor of every
modern on-policy method.

### 10.2 The n-Step Advantage and GAE

Between the one-step TD error and the full return lies a family of
advantage estimates.  Define the $`k`$-step TD error:

$$
\large
\delta_t^{(k)}
= -V_\phi(s_t) + R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{k-1} R_{t+k} + \gamma^k V_\phi(s_{t+k}).
$$

**Generalized Advantage Estimation** (Schulman et al. 2016, [9]) is the
exponentially weighted average of all $`k`$-step advantages, with weight
$`(1 - \lambda)\\, \lambda^{k-1}`$ for a parameter $`\lambda \in [0,1]`$:

$$
\large
A_t^{\text{GAE}(\gamma, \lambda)}
= \sum_{k=1}^{\infty} (1-\lambda)\\, \lambda^{k-1}\\,
  \Bigl( \delta_t^{(k)} \Bigr)
= \sum_{k=0}^{\infty} (\gamma \lambda)^k\\,
  \Bigl( R_{t+1} + \gamma\\, V_\phi(s_{t+k+1}) - V_\phi(s_{t+k}) \Bigr).
$$

**Proof of the telescoping form.**  Let $`\delta_t = R_{t+1} + \gamma V_\phi
(s_{t+1}) - V_\phi(s_t)`$ be the one-step TD error.  A direct calculation
shows

$$
\large
\delta_t^{(k)}
= \sum_{k'=0}^{k-1} \gamma^{k'}\\, \delta_{t+k'}.
$$

Indeed both sides telescope: the intermediate $`\gamma V_\phi`$ terms cancel
pairwise, leaving $`-V_\phi(s_t) + \sum_{j=0}^{k-1} \gamma^j R_{t+1+j} +
\gamma^k V_\phi(s_{t+k})`$ = the definition.  Substituting into the weighted
average and exchanging sums,

$$
\large
A_t^{\text{GAE}} = \sum_{k=1}^{\infty} (1-\lambda)\lambda^{k-1}
 \sum_{k'=0}^{k-1} \gamma^{k'} \delta_{t+k'}
= \sum_{k'=0}^{\infty} (\gamma\lambda)^{k'}\\, \delta_{t+k'},
$$

where the last equality is a standard change of summation order
(collecting the coefficient of $`\delta_{t+k'}`$; the inner geometric series
over $`k \ge k'+1`$ sums to
$`(1-\lambda)\lambda^{k'} / (1-\lambda) = \lambda^{k'}`$, and multiplying by
$`\gamma^{k'}`$ gives $`(\gamma\lambda)^{k'}`$).  The result:

$$
\large
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma\lambda)^k\\,
\delta_{t+k},
\qquad
\delta_t = R_{t+1} + \gamma\\, V_\phi(s_{t+1}) - V_\phi(s_t).
$$

**Interpretation.**  $`\lambda = 0`$ recovers one-step TD (high bias, low
variance); $`\lambda \to 1`$ recovers the Monte-Carlo return minus baseline
(low bias, high variance).  GAE interpolates with one parameter, exactly as
the $`\lambda`$-return interpolates for values (Section 4.5).  PPO uses
$`\lambda \approx 0.95`$ and estimates the advantage by *bootstrapping the
critic once per rollout, then summing with weights $`(\gamma\lambda)^k`$.

In code this is computed with the *recurrent* form
$`A_t = \delta_t + \gamma \lambda (1-d_t)\\, A_{t+1}`$ evaluated backwards
from the end of a rollout — the numerically stable version of the
infinite sum above.

---

## 11. Trust Regions And The Clipped Surrogate (PPO)

### 11.1 Why "Small Policy Steps"

If the policy gradient step is large, the new policy $`\pi_{\theta'}`$ can
move far from the old $`\pi_\theta`$ — and the advantage estimates were
computed under $`\pi_\theta`$, so they become stale.  The TRPO objective
(Schulman et al. 2015, [10]) enforces a hard constraint on the KL divergence
between successive policies; PPO (Schulman et al. 2017, [11]) replaces the
hard constraint with a clipped objective that is cheap and well-behaved
under SGD.

### 11.2 Importance Sampling for the Policy Ratio

The probability-ratio of the new policy to the old one, evaluated at the
sampled actions, corrects for the fact that data was collected under
$`\pi_{\theta_\text{old}}`$:

$$
\large
r_t(\theta) \\;=\\;
\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_\text{old}}(A_t \mid S_t)}.
$$

**Derivation: why the ratio appears.**  The surrogate objective is

$$
\large
L^{\pi}(\theta)
= \mathbb{E}_{s \sim d^{\pi_{\text{old}}},\\, a \sim \pi_{\text{old}}}\\!\left[
  \frac{\pi_\theta(a \mid s)}{\pi_{\text{old}}(a \mid s)}\\, A^{\pi_{\text{old}}}(s,a)
\right],
$$

which is a first-order (in the policy) approximation of the true objective
$`J(\theta)`$: at $`\theta = \theta_{\text{old}}`$ the ratio is $`1`$, so
the surrogate and its gradient agree with the true objective to first
order.
The importance-weight appears because the expectation is over old-policy
trajectories but we want to evaluate the new policy — the same
score-function/change-of-measure idea as Section 9.2.

### 11.3 The Clipped Surrogate

PPO's objective:

$$
\large
L^{\text{CLIP}}(\theta)
= \mathbb{E}_t\\!\left[
  \min\\!\Bigl(\\,
    r_t(\theta)\\, A_t,\\;
    \mathrm{clip}(r_t(\theta),\\; 1-\varepsilon,\\; 1+\varepsilon)\\, A_t
  \Bigr) \right].
$$

**Why the min does what we want.**  When $`A_t > 0`$ (the action was better
than average), the objective wants $`r_t(\theta)`$ to be large — but only up
to $`1 + \varepsilon`$; beyond that the clipped term is constant, so the min
*pushes the gradient to zero* rather than encouraging the policy to change
too much.  When $`A_t < 0`$ (the action was worse), the objective wants
$`r_t`$ small — the min *keeps it from collapsing toward $`0`$ unboundedly
(the floor $`1 - \varepsilon`$).  In both cases the surrogate is a
lower bound
of the unclipped objective, and the bound is tight within the clip
range — PPO maximizes a conservative lower bound, the practical analogue of
TRPO's hard KL constraint.  The resulting gradient (by the chain rule and
the ratio definition) is

$$
\large
A_t\\,
\begin{cases}
\nabla_\theta r_t(\theta), & r_t(\theta) \in (1-\varepsilon,\\; 1+\varepsilon)
  \\;\text{or}\\; (A_t < 0, r_t(\theta) > 1+\varepsilon) \\
0, & \text{otherwise (clipped)}.
\end{cases}
$$

### 11.4 The Full PPO Objective

The complete PPO loss adds a value-function loss and an entropy bonus:

$$
\large
L^{\text{PPO}}(\theta) = \mathbb{E}_t\\!\left[
  L^{\text{CLIP}}(\theta) - c_1\\, L^{\text{VF}}(\theta) + c_2\\, H(\pi_\theta)(s_t)
\right],
$$

where $`L^{\text{VF}}`$ is (typically clipped) MSE between the critic and the
GAE-computed returns, and $`H`$ is the policy entropy (exploration bonus;
Section 9.1 intuition).  The entropy term is

$$
\large
H(\pi_\theta)(s) = -\sum_a \pi_\theta(a \mid s)\\, \log \pi_\theta(a \mid s),
$$

which is maximized by the uniform policy — the natural complement to the
greedy signal: it keeps probability mass spread while the value signal
concentrates it.

In practice PPO collects rollouts, computes GAE advantages with the critic,
then runs several SGD epochs over the *same* rollouts with the clipped
surrogate (Section 11.3), a value loss, and an entropy bonus.

---

# Part 5 — Maximum-Entropy RL And Soft Actor-Critic

## 12. Soft Value Functions And Soft Bellman Equations

### 12.1 The Maximum-Entropy Objective

Classic RL maximizes expected return $`\sum \mathbb{E}[r]`.
**Maximum entropy RL** (Ziebart et al. 2008; Haarnoja et al. 2017, [12])
augments the return with the policy's entropy at every visited state:

$$
\large
J(\theta) \\;=\\;
\sum_{t} \mathbb{E}_{(s_t, a_t) \sim \pi_\theta}\\!\left[
  r(s_t, a_t) + \alpha\\, \mathcal{H}\\!\left(\pi_\theta(\cdot \mid s_t)\right)
\right],
$$

where $`\alpha > 0`$ trades reward vs. exploration (the **temperature**).
Maximizing entropy keeps the policy stochastic and exploratory, which (a)
improves exploration, (b) makes the policy robust to model error, and
(c) can be warmer-started toward multi-modal behaviors.  SAC makes
$`\alpha`$ an *adaptive, learned* parameter (Section 13.4).

### 12.2 Soft State- and Action-Value Functions

Define the **soft Q-function** and **soft V-function** with an entropy term
built into the value:

$$
\large
Q_{\text{soft}}^{\pi}(s, a)
= r(s, a) + \gamma\\, \mathbb{E}_{s' \sim p}\\!\left[ V_{\text{soft}}^{\pi}(s') \right],
\qquad
V_{\text{soft}}^{\pi}(s)
= \mathbb{E}_{a \sim \pi}\\!\left[ Q_{\text{soft}}^{\pi}(s, a) - \alpha\\, \log \pi(a \mid s) \right].
$$

The term $`-\alpha \log \pi(a \mid s)`$ is the *entropy bonus in expectation*:
it is exactly $`\alpha \mathcal{H}(\pi(\cdot \mid s))`$ when averaged over
$`a \sim \pi`$.  (Since $`\mathcal{H}(\pi) = -\mathbb{E}[\log \pi]`$.)

### 12.3 The Soft Bellman Equation

Combining the two definitions gives the **soft Bellman equation**:

$$
\large
Q_{\text{soft}}^{\pi}(s, a)
= r(s, a) + \gamma\\, \mathbb{E}_{s'}\\!\left[
  \mathbb{E}_{a' \sim \pi}\\!\left[
    Q_{\text{soft}}^{\pi}(s', a') - \alpha\\, \log \pi(a' \mid s')
  \right]
\right].
$$

If we write the inner expectation explicitly as a sum, it reads

$$
\large
Q_{\text{soft}}^{\pi}(s, a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s, a)\\,
\sum_{a'} \pi(a' \mid s')\\,
\Bigl[ Q_{\text{soft}}^{\pi}(s', a') - \alpha \log \pi(a' \mid s') \Bigr],
$$

which is the *soft analogue* of the classic Bellman expectation equation
(Section 3.2) with the reward augmented by the entropy bonus.  The soft
Bellman operator is still a $`\gamma`$-contraction in the sup-norm — the
proof is identical to Section 3.4 (and the fixed-point consequences follow
from the Banach theorem proved in Section 0.3) since the entropy term is a
deterministic function of $`s'`$ and adds no randomness to the contraction
argument.

### 12.4 The Optimal Soft Policy and "Softmax" Form

**Claim (soft policy improvement).**  The policy that maximizes the
soft value satisfies

$$
\large
\pi^{\ast}_{\text{soft}}(a \mid s)
\\;\propto\\;
\exp\\!\left( \frac{1}{\alpha}\\, Q_{\text{soft}}^{\ast}(s, a) \right).
$$

**Proof.**  At each state $`s`$, the soft value as a function of the policy
is

$$
\large
V_{\text{soft}}^{\pi}(s) =
    \mathbb{E}_{a \sim \pi}\\!\left[\\, Q_{\text{soft}}^{\pi}(s, a) - \alpha \log \pi(a \mid s)
    \\,\right].
$$

This is exactly the entropy-regularized objective of Lemma 0.4 with
$`Q(a) = Q_{\text{soft}}^{\pi}(s, a)`$ acting on the finite action set at
state $`s`$.  By that lemma, the unique maximizer over the action simplex
is the Gibbs distribution

$$
\large
\pi^{\ast}(a \mid s) \\;=\\; \frac{\exp\\!\left( Q_{\text{soft}}^{\pi}(s,a) /
\alpha \right)}{\sum_{a'} \exp\\!\left( Q_{\text{soft}}^{\pi}(s,a') / \alpha
\right)}.
$$

Writing $`\pi^{\ast}`$ as the *improvement* of $`\pi`$, this holds for any
$`\pi`$, in particular for the optimal $`\pi^{\ast}`$ itself; substituting
$`Q_{\text{soft}}^{\ast}`$ for $`Q_{\text{soft}}^{\pi}`$ in the argument
gives the claimed form.  The Lagrange-multiplier derivation (concavity,
first-order condition, normalization) is carried out in full in Lemma 0.4.
QED

**Consequence.**  In maximum-entropy RL, the *optimal* policy is stochastic,
proportional to exponentiated Q — never a hard arg-max.  This is the deep
reason SAC uses a parameterized, reparameterizable policy rather than a
greedy one; and it is why the soft target in Section 13.3 integrates
over $`a' \sim \pi`$ with the entropy bonus rather than taking a max.

---

## 13. Soft Actor-Critic

### 13.1 The SAC Components

SAC (Haarnoja et al. 2018, [13]) is an off-policy maximum-entropy
actor-critic with three learned components:

- a **policy** $`\pi_\phi(a \mid s)`$ (the *actor*), parameterized so that
  actions can be sampled via a differentiable reparameterization
  $`a = f_\phi(s, \xi)`$, $`\xi \sim \mathcal{N}(0, I)`$;
- a **soft Q-function** $`Q_\psi(s, a)`$ (the *critic*);
- a **target soft Q-function** $`Q_{\bar\psi}(s, a)`$, a lagged copy
  (Polyak-averaged), exactly as in Section 6.5.

SAC trains the Q-function by the soft Bellman residual, and the policy by
minimizing the KL-divergence between the policy and the soft-optimal Gibbs
policy of Section 12.4.

### 13.2 The Critic Loss (Soft Q-Learning)

The critic minimizes the mean squared soft Bellman residual over replay
transitions $`(s, a, r, s', d)`$:

$$
\large
\mathcal{L}_Q(\psi) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}}\\!\left[
  \Bigl( Q_\psi(s, a) - y \Bigr)^2
\right],
\qquad
y = r + \gamma (1-d)\\,
\Bigl[
  \min_{j=1,2} Q_{\bar\psi_j}(s', \tilde{a}') - \alpha\\, \log \pi_\phi(\tilde{a}' \mid s')
  \Bigr],
$$

where $`\tilde{a}' \sim \pi_\phi(\cdot \mid s')`$ is a *freshly
reparameterized* sample of the current policy at the next state.  The
$`\min`$ over two target critics (twin critics, following TD3) curbs
maximization bias (Section 7.1) in the off-policy setting.  The entropy
bonus $`-\alpha \log \pi_\phi(\tilde{a}' \mid s')`$ implements the soft
Bellman equation of Section 12.3 — the target is *soft*, not greedy.

**Why the target uses the min and the entropy bonus together.**  The soft
value of the *current policy* is
$`\mathbb{E}_{a'}[Q(s',a') - \alpha \log \pi(a' \mid s')]`$ (Section 12.2).
Replacing the expectation by a single sample $`\tilde{a}'`$ gives an unbiased
estimate; the *min over two targets* reduces the positive bias from taking a
sample of the max-like quantity (the twin-critic analogue of double-DQN's
argument in Section 7.2).

### 13.3 The Actor Loss (KL to the Soft-Optimal Policy)

The actor minimizes the KL divergence from the soft-optimal Gibbs policy:

$$
\large
\mathcal{L}_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\\!\left[
  \mathbb{E}_{\tilde{a} \sim \pi_\phi(\cdot \mid s)}\\!\left[
    \alpha \log \pi_\phi(\tilde{a} \mid s) - Q_\psi(s, \tilde{a})
  \right]
\right].
$$

**Derivation.**  The soft-optimal (Gibbs) policy at state $`s`$ is
$`\pi^{\ast} \propto \exp(Q_\psi(s, \cdot)/\alpha)`$ (Section 12.4).  Minimizing
$`\mathrm{KL}(\pi_\phi \\,\|\\, \pi^{\ast})`$ over $`\pi_\phi`$:

$$
\large
\mathrm{KL}(\pi_\phi \\,\|\\, \pi^{\ast})
= \mathbb{E}_{a \sim \pi_\phi}\\!\left[
  \log \pi_\phi(a \mid s) - \log \pi^{\ast}(a \mid s)
\right].
$$

Substituting $`\log \pi^{\ast}(a \mid s) = Q_\psi(s,a)/\alpha - \log Z(s)`$ (where
$`Z(s)`$ is the normalizer, constant in $`\phi`$), dropping the constant,
and multiplying by $`\alpha`$ yields exactly $`\mathcal{L}_\pi`$.  The
expectation is over the *policy's own* samples: differentiating it requires
a gradient of an expectation under the distribution being optimised, which
is exactly what the reparameterization trick provides.

**Derivation (reparameterized gradient).**  Write the action as a
deterministic function of the state and a noise variable,
$`a = f_\phi(s, \xi)`$ with $`\xi \sim \mathcal{N}(0,I)`$, where $`f_\phi`$
is fixed and differentiable in $`\phi`$.  Because the law of $`\xi`$ does
not depend on $`\phi`$, the expectation is over a fixed distribution, so by
differentiating under the integral,

$$
\large
\nabla_\phi\\,
\mathbb{E}_{a \sim \pi_\phi}[\\, f(a) \\,]
= \nabla_\phi\\, \mathbb{E}_\xi[\\, f(f_\phi(s, \xi)) \\,]
= \mathbb{E}_\xi\\!\left[\\, \nabla_\phi f(f_\phi(s, \xi)) \\,\right].
$$

Contrast the score-function estimator of Section 9.2:
$`\nabla_\phi \mathbb{E}_{\pi_\phi}[f] = \mathbb{E}_{\pi_\phi}[\\, f\\,
\nabla_\phi \log \pi_\phi \\,]`$.  Both estimators are unbiased.  The score
estimator multiplies $`f(a)`$ by the score
$`\nabla_\phi \log \pi_\phi(a \mid s)`$, whose magnitude varies strongly
from action to action, inflating the variance of any Monte-Carlo average;
the reparameterized estimator differentiates $`f`$ along the smooth path
$`f_\phi`$ instead, so each sample contributes the gradient of the
objective itself.  This is the standard score-free trick that makes the
whole loss a plain SGD problem — no high-variance score-function estimator
needed — and it is exactly why SAC parameterizes the policy as
$`f_\phi(s, \xi)`$.

### 13.4 The Adaptive Temperature

The temperature $`\alpha`$ is learned by dual gradient descent to enforce a
target entropy constraint $`\mathcal{H}_0`$ (SAC default: $`-\dim(\mathcal{A})`$):

$$
\large
\mathcal{L}(\alpha) = -\alpha\\, \mathbb{E}_{a \sim \pi_\phi}\\!\left[
  \log \pi_\phi(a \mid s) + \mathcal{H}_0
\right].
$$

**Derivation.**  The constrained problem "maximize return subject to
$`\mathcal{H}(\pi_\phi(\cdot \mid s)) \ge \mathcal{H}_0`$ for every $`s`$"
has Lagrangian $`\mathbb{E}[r] + \alpha(\mathcal{H} - \mathcal{H}_0)`$ with
multiplier $`\alpha \ge 0`$.  Minimizing the Lagrangian wrt $`\alpha`$ gives
the loss above; the gradient of $`\mathcal{L}(\alpha)`$ pushes $`\alpha`$ up
when entropy is below target and down when above — an automatic
exploration/exploitation balance.  This is why SAC typically needs no
hand-tuned $`\varepsilon`$ schedule (unlike DQN, Section 6.6).

In practice SAC implements all four losses — critic (twin soft Q),
actor (KL), temperature (dual), and the target-critic Polyak update —
as separate optimization groups or a single joint loss.

---

# Appendix A: Symbol Table

Every symbol used in this document, with a plain-English meaning and the
section where it first appears.

| Symbol | Meaning | First appears |
|---|---|---|
| $`\mathcal{S}, \mathcal{A}`$ | state space, action space | §2.1 |
| $`s, a, s'`$ | a state, an action, a next state | §2.1 |
| $`S_t, A_t, R_{t+1}`$ | state, action, reward at time $`t`$ (reward after $`A_t`$) | §1 |
| $`p(s' \mid s, a)`$ | transition kernel | §2.1 |
| $`r(s, a, s')`$ | reward function | §2.1 |
| $`\gamma`$ | discount factor, $`\gamma \in [0,1)`$ | §2.1 |
| $`\pi(a \mid s)`$ | policy (conditional action distribution) | §2.3 |
| $`\pi_\theta`$ | parameterized policy | §9.1 |
| $`\Pr_\pi, \mathbb{E}_\pi`$ | probability/expectation under $`\pi`$ | §2.3 |
| $`G_t`$ | return from time $`t`$ | §2.4 |
| $`V^{\pi}(s), Q^{\pi}(s,a)`$ | state-/action-value of $`\pi`$ | §3.1 |
| $`V^{\ast}, Q^{\ast}`$ | optimal value functions | §3.1 |
| $`\mathcal{T}^\pi`$ | Bellman (evaluation) operator | §3.4 |
| $`\lVert \cdot \rVert_\infty`$ | sup-norm | §3.4 |
| $`\alpha`$ | learning rate (or step size); SAC temperature | §4.2, §12.1 |
| $`\delta_t`$ | TD error | §4.3 |
| $`d_t`$ | done flag (1 iff terminal) | §4.4 |
| $`G_t^{(n)}`$ | n-step return | §4.5 |
| $`\varepsilon`$ | exploration probability (epsilon-greedy), PPO clip width | §5.1, §11.3 |
| $`Q_\theta`$ | Q-network with weights $`\theta`$ | §6.1 |
| $`\theta^-`$ | target-network weights | §6.2 |
| $`C`$ | hard target-update frequency | §6.5 |
| $`\tau`$ | Polyak soft-update coefficient | §6.5 |
| $`A(s,a)`$ | advantage | §8.1 |
| $`\mathrm{clip}(\cdot, a, b)`$ | clamp to $`[a, b]`$ | §11.3 |
| $`r_t(\theta)`$ | importance ratio, $`\pi_\theta / \pi_{\text{old}}`$ | §11.2 |
| $`\lambda`$ | GAE trace-decay / return-mixing parameter | §10.2 |
| $`\mathcal{H}(\pi)`$ | entropy of $`\pi`$ | §11.4 |
| $`\alpha`$ (SAC) | temperature | §12.1 |
| $`Q_{\text{soft}}, V_{\text{soft}}`$ | soft values | §12.2 |
| $`\psi, \phi, \bar\psi`$ | critic, actor, target-critic weights (SAC) | §13.1 |
| $`\xi`$ | reparameterization noise | §13.1 |
| $`\mathcal{D}`$ | replay buffer (distribution) | §13.2 |
| $`\mathcal{H}_0`$ | target entropy | §13.4 |
| $`\mathrm{KL}(\cdot \| \cdot)`$ | Kullback–Leibler divergence | §13.3 |
| $`Z(s)`$ | Gibbs normalizer | §13.3 |

---

# Appendix B: Failure Modes And Shortcuts

**B.1 Missing `done` masking.**  Forgetting the $`(1-d)`$ factor in the TD
target (Sections 4.4, 6.2) lets the value of terminal states bootstrap into
non-existent future reward — the most common RL correctness bug.

**B.2 Overestimation from the `max`.**  Section 7.1 proves the upward bias;
Section 7.2 gives the Double-DQN fix.  Watch for value metrics creeping
above the true optimal return — a signature of unaddressed maximization bias.

**B.3 Non-stationary target.**  Updating $`\theta`$ against a target built
from $`\theta`$ itself (Section 6.3.2) causes oscillation/divergence; the
target network (Section 6.4) exists precisely to remove this.

**B.4 Correlated batches.**  Training on consecutive transitions
(Section 6.3.1) over-fits recent experience; replay fixes it.  Symptoms:
fast early learning then collapse, or catastrophic forgetting of early
discoveries.

**B.5 Policy–data feedback / online-only data.**  Without a replay buffer
(Section 6.3.3), the improving policy changes the data distribution and
early experience is lost.  On-policy methods (PPO) accept this by design and
use short rollouts + multiple epochs (Section 11.4) instead.

**B.6 Pathological exploration schedule.**  Epsilon decaying too fast (DQN)
or temperature collapsing (SAC) locks in premature exploitation
(Sections 6.6, 13.4).  A resumed run should continue the schedule from the
checkpointed step, not restart it.

**B.7 Learning step with env side effects.**  The learning update must be a
pure function of a batch of transitions.  If it also steps the environment,
the update is no longer testable in isolation and the same experience is
used twice (once to learn, once to drive the env), corrupting the i.i.d.
assumption behind the loss.

**B.7.1 Replay sampling from an empty buffer.**  Sampling before the buffer
is warm (filled with enough random experience) returns tiny/correlated
batches; practical systems warm the buffer first and raise a clear error if
a batch is requested too early.

**B.8 Rewards not normalized.**  Large reward scales destabilize value
learning (a large $`\gamma`$ compounds them; Section 2.4).  Normalize or
clip rewards inside the environment's reward transform, and document it in
the environment contract.

**B.9 Unstable critic early on.**  The critic (actors' baseline) is wrong
early; GAE with $`\lambda`$ interpolation (Section 10.2) and value clipping
(PPO, Section 11.4) bound the damage.

---

# Appendix C: Reading List

The methods in this document come from a long and well-documented line of
work.  Canonical references, each paired with where it is used here; every
entry is a public, checkable source.

**Foundational theory**

- [1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An
  Introduction*, 2nd ed., MIT Press, 2018.  Parts 1–2 (Bellman equations,
  TD, Q-learning) follow this book closely.
  URL: http://incompleteideas.net/book/the-book-2nd.html
- [2] C. J. C. H. Watkins and P. Dayan, *Q-Learning*, Machine Learning 8,
  1992.  §5.2 (Q-learning convergence).
  URL: https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
- [3] T. Jaakkola, M. I. Jordan, and S. P. Singh, *On the Convergence of
  Stochastic Iterative Dynamic Programming Algorithms*, Neural Computation 6,
  1994.  §5.2 (convergence of stochastic approximation of the Bellman
  optimality operator).
  URL: https://direct.mit.edu/neco/article/6/6/1185/5826/On-the-Convergence-of-Stochastic-Iterative-Dynamic

**Deep RL**

- [4] V. Mnih et al., *Human-level control through deep reinforcement
  learning*, Nature 518, 2015.  §6 (DQN, replay, target network).
  URL: https://www.nature.com/articles/nature14236
- [5] T. Schaul, J. Quan, I. Antonoglou, D. Silver, *Prioritized Experience
  Replay*, ICLR 2016.  §6.3 (prioritized replay).
  URL: https://arxiv.org/abs/1511.05952
- [6] H. van Hasselt, *Double Q-learning*, NeurIPS 2010.  §7 (maximization
  bias).
  URL: https://papers.nips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
- [7] Z. Wang et al., *Dueling Network Architectures for Deep Reinforcement
  Learning*, ICML 2016.  §8 (advantage decomposition).
  URL: https://arxiv.org/abs/1511.06581

**Policy gradients and PPO**

- [8] R. S. Sutton, D. McAllester, S. Singh, Y. Mansour, *Policy Gradient
  Methods for Reinforcement Learning with Function Approximation*, NeurIPS
  1999.  §9 (policy gradient theorem).
  URL: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- [9] J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel,
  *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation*, ICLR 2016.  §10.2 (GAE).
  URL: https://arxiv.org/abs/1506.02438
- [10] J. Schulman et al., *Trust Region Policy Optimization*, ICML 2015.
  §11.1 (trust regions, KL constraint).
  URL: https://arxiv.org/abs/1502.05477
- [11] J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
  §11 (clipped surrogate).
  URL: https://arxiv.org/abs/1707.06347

**Maximum-entropy RL and SAC**

- [12] B. Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning*,
  AAAI 2008; T. Haarnoja et al., *Reinforcement Learning with Deep Energy
  Based Policies*, ICML 2017.  §12 (soft Bellman and Gibbs policy).
  URL: https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf;
  URL: https://arxiv.org/abs/1702.08165
- [13] T. Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep Reinforcement Learning with a Stochastic Actor*, ICML 2018.  §13
  (SAC: critic, actor, temperature).
  URL: https://arxiv.org/abs/1801.01290
- [14] S. Fujimoto, H. van Hoof, D. Meger, *Addressing Function
  Approximation Error in Actor-Critic Methods* (TD3), ICML 2018.  §13.2
  (twin critics, target policy smoothing).
  URL: https://arxiv.org/abs/1802.09477

**Software documentation**

- Gymnasium API reference — `gym.Env` (reset/step/close), the standard
  environment protocol used throughout this document.
- PyTorch docs — `torch.utils.data.DataLoader`, `torch.amp.GradScaler`,
  `torch.nn.utils.clip_grad_norm_` (PPO value clipping/optimization).