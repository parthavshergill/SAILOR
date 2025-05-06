# SAILOR Project Implementation Plan

**Project Goal:** Implement and evaluate SAILOR (Self-Alignment for LOng horizon Rewards), a system using LLM-driven self-alignment to learn effective reward functions for an RL agent playing the game 2048, focusing on long-term strategy. We will also evaluate novel improvements: constrained reward editing and weighted preference ranking.

**Target Environment:** 2048 (via an open-source Gym/Gymnasium implementation)
**Core Technique:** Self-Alignment based on Zeng et al. (2024), adapted for 2048.
**RL Algorithm:** Proximal Policy Optimization (PPO) as the primary algorithm (using Stable-Baselines3).
**LLM:** Open-source model via Hugging Face `transformers` (e.g., Mistral-7B, Llama-3-8B, Phi-3-Mini).
**Optimization:** Bayesian Optimization for reward function updates (`scikit-optimize`).

---

## Core Libraries & Initial Setup

**Goal:** Establish the foundational software environment.

**Libraries to Install:**
* `gymnasium`: For the RL environment interface.
* `stable-baselines3[extra]`: For the PPO RL algorithm and utilities. Include `[extra]` for features like TensorBoard logging.
* `transformers`: For loading and using Hugging Face LLMs.
* `torch`: The backend deep learning framework for SB3 and Transformers. Ensure compatibility.
* `scikit-optimize`: For Bayesian Optimization.
* `numpy`: For numerical operations.
* `pandas`: Potentially useful for handling trajectory data or results.
* `matplotlib`/`seaborn`: For plotting results.
* A `gymnasium-2048` implementation (find one on PyPI/GitHub, e.g., search for one compatible with `gymnasium`).

**Setup Actions:**
1.  Create a virtual environment (e.g., using `conda` or `venv`).
2.  Install the libraries listed above (`pip install ...`).
3.  Set up a Git repository for the project and push initial structure.
4.  Ensure AWS accounts are active and CLI access is configured (if running on AWS).

---

## Phase 1: Environment and Baseline Agent

**(Corresponds roughly to Weeks 1-2)**

### Step 1.1: Environment Integration

* **Goal:** Load and wrap the chosen 2048 Gymnasium environment for use with Stable-Baselines3.
* **Input:** Selected `gymnasium-2048` library.
* **Output:** A Gymnasium environment instance (`env`) that conforms to the standard API (`step`, `reset`, `observation_space`, `action_space`).
* **Key Libraries/Functions:** `gymnasium.make()`, potentially `gymnasium.Wrapper` if customization is needed (e.g., observation normalization, action masking if the base env doesn't provide it well).
* **Implementation Details:**
    * Instantiate the environment: `env = gymnasium.make('Gymnasium2048-v0')` (replace with the actual registered ID).
    * Verify `observation_space` (likely a `Box` representing the grid) and `action_space` (likely `Discrete(4)` for up/down/left/right).
    * Implement a simple random agent loop (`env.reset()`, `env.step(env.action_space.sample())`) to test basic interaction and rendering (`env.render()`).
    * Consider using `stable_baselines3.common.env_checker.check_env(env)` to ensure compatibility.

### Step 1.2: Baseline RL Agent (PPO)

* **Goal:** Implement and train a standard PPO agent on the 2048 environment using a basic reward function.
* **Input:** The integrated Gymnasium environment (`env`).
* **Output:** A trained PPO model (`model`) saved to disk. Training logs (e.g., TensorBoard).
* **Key Libraries/Functions:** `stable_baselines3.PPO`, `stable_baselines3.common.env_util.make_vec_env`, `model.learn()`, `model.save()`.
* **Implementation Details:**
    * Define the baseline reward function (see Step 1.3) - initially, this might be integrated directly into the environment wrapper or passed during training.
    * Instantiate the PPO model: `model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_2048_tensorboard/")`. Use `MlpPolicy` assuming a flat observation space; adjust if using CNNs for grid observation.
    * Train the model: `model.learn(total_timesteps=1000000)` (adjust timesteps based on performance/budget). Monitor training using TensorBoard.
    * Save the trained model: `model.save("ppo_2048_baseline")`.
    * Evaluate the baseline model periodically (e.g., using `stable_baselines3.common.evaluation.evaluate_policy`) to establish baseline performance metrics.

### Step 1.3: Baseline Reward Function

* **Goal:** Define the initial, simple reward function for the baseline agent. This function will later be replaced by the learnable $R_{\theta}$.
* **Input:** Game state information from the environment `step` function (e.g., score change, board state).
* **Output:** A scalar reward value for a given transition.
* **Implementation Details:**
    * Implement a function `calculate_baseline_reward(info, old_score, new_score, board_state, done)`:
        * **Primary Reward:** `reward = new_score - old_score` (reward for merging tiles).
        * **Optional Shaping (Simple):** Add a small bonus/penalty based on simple heuristics, e.g.:
            * Bonus for keeping the highest tile in a corner.
            * Penalty for game over (`done`).
            * (Keep it simple initially, as the goal is for the LLM to learn better strategies).
    * Integrate this logic into the environment wrapper's `step` method or pass it to the RL training process if the library allows custom reward calculation.

---

## Phase 2: Core Self-Alignment Pipeline

**(Corresponds roughly to Weeks 3-6)**

### Step 2.1: LLM Setup

* **Goal:** Load a pre-trained open-source LLM and set up a function to query it for trajectory ranking.
* **Input:** Chosen LLM model name/path (e.g., `"mistralai/Mistral-7B-Instruct-v0.1"`).
* **Output:** A function `rank_trajectories_llm(trajectories)` that takes a list of trajectories and returns ranked indices or scores.
* **Key Libraries/Functions:** `transformers.AutoTokenizer`, `transformers.AutoModelForCausalLM`, `model.generate()`.
* **Implementation Details:**
    * Load tokenizer and model:
        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Or other chosen model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
            device_map="auto" # Automatically use GPU if available
        )
        ```
    * Define a prompt template for the LLM. This is critical. It needs to instruct the LLM to evaluate trajectories based on strategic play towards winning 2048, considering long-term positioning, not just immediate score gains. Include Chain-of-Thought elements if desired. Example snippet:
        ```prompt
        You are an expert 2048 player. Analyze the following game trajectories. A trajectory consists of a sequence of (board state, action, next board state, score increase). Rank these trajectories based on which one demonstrates better strategic planning towards achieving a high score and ultimately the 2048 tile. Consider factors like maintaining open space, building large tiles in corners, setting up merges, and avoiding moves that lock the board. Explain your reasoning step-by-step (Chain of Thought) before providing the final ranking.

        Trajectory 1:
        [State 1_1, Action 1_1, State 1_2, Score Inc 1_1]
        [State 1_2, Action 1_2, State 1_3, Score Inc 1_2]
        ...

        Trajectory 2:
        [State 2_1, Action 2_1, State 2_2, Score Inc 2_1]
        ...

        [Your Chain-of-Thought Analysis Here]

        Final Ranking (most strategic first): [Trajectory_Index_1, Trajectory_Index_2, ...]
        ```
    * Create the `rank_trajectories_llm` function:
        * Input: A list of trajectories. Each trajectory could be a list of tuples `(observation, action, reward, next_observation, done, info)`.
        * Format the trajectories into the prompt. Represent board states clearly (e.g., formatted numpy arrays).
        * Tokenize the prompt.
        * Generate response using `model.generate()`. Adjust generation parameters (`max_new_tokens`, `temperature`, `do_sample`).
        * Parse the LLM's response to extract the ranked order of trajectory indices. Handle potential parsing errors.
        * Consider batching prompts if querying the LLM for many trajectories simultaneously, if the model/hardware supports it.
        * **Budget Consideration:** LLM inference is computationally expensive. Optimize prompt length, use efficient model loading (e.g., `bfloat16`, quantization if needed), and potentially limit the number/length of trajectories ranked per iteration.

### Step 2.2: Parameterized Reward Function Definition ($R_{\theta}$)

* **Goal:** Define a reward function whose behavior is controlled by a set of learnable parameters $\theta$.
* **Input:** Game state information (board state, score change, potentially derived features).
* **Output:** A function `calculate_parametric_reward(theta, state, action, next_state, score_change, done)` that returns a scalar reward. A vector $\theta$ representing the initial parameters.
* **Implementation Details:**
    * Define the structure of $R_{\theta}$. It should compute features from the state/transition and combine them using the parameters $\theta$.
    * Example Features:
        * `f_score_increase`: `new_score - old_score`
        * `f_max_tile_corner`: 1 if max tile in corner, 0 otherwise.
        * `f_empty_tiles`: Number of empty tiles.
        * `f_monotonicity`: Measure of how monotonic rows/columns are (encourages smooth gradients).
        * `f_merges_available`: Number of possible merges in the next state.
    * Combine features using parameters $\theta$:
        `reward = theta[0]*f_score_increase + theta[1]*f_max_tile_corner + theta[2]*f_empty_tiles + ...`
    * The function should take `theta` as an argument.
    * Initialize `theta` (e.g., `theta_initial = np.array([1.0, 0.1, 0.01, ...])`). The size of `theta` depends on the number of features chosen.

### Step 2.3: Trajectory Sampling

* **Goal:** Collect trajectories using the current RL agent's policy $\pi_{\theta}$ (trained with the *current* $R_{\theta}$).
* **Input:** Trained RL agent (`model`), environment instance (`env`), number of trajectories `M`.
* **Output:** A list of `M` trajectories. Each trajectory is a list of transition tuples `(obs, action, reward, next_obs, done, info)`.
* **Key Libraries/Functions:** `model.predict()`, `env.step()`, `env.reset()`.
* **Implementation Details:**
    * Implement a function `sample_trajectories(model, env, num_trajectories)`:
        * Loop `num_trajectories` times:
            * Initialize `current_trajectory = []`.
            * `obs, info = env.reset()`.
            * `done = False`.
            * While `not done`:
                * `action, _states = model.predict(obs, deterministic=False)` (use stochastic policy for exploration).
                * `next_obs, reward, terminated, truncated, info = env.step(action)` (Note: use the *current* $R_{\theta}$ to calculate this `reward` for consistency within the trajectory data, though it won't be directly used for ranking comparison later).
                * `done = terminated or truncated`.
                * Store `(obs, action, reward, next_obs, done, info)` in `current_trajectory`.
                * `obs = next_obs`.
            * Add `current_trajectory` to the list of trajectories.
    * Return the list of trajectories.

### Step 2.4: Reward Function ($R_{\theta}$) Ranking

* **Goal:** Rank the sampled trajectories based *only* on the sum of rewards calculated using the current parameterized reward function $R_{\theta}$.
* **Input:** List of trajectories, current reward parameters `theta`, the `calculate_parametric_reward` function.
* **Output:** A list of trajectory indices, sorted from highest total $R_{\theta}$ reward to lowest.
* **Implementation Details:**
    * Implement a function `rank_trajectories_r_theta(trajectories, theta, reward_function)`:
        * For each trajectory `traj` in `trajectories`:
            * `total_reward = 0`.
            * For each transition `(obs, act, _, next_obs, done, info)` in `traj`:
                * Extract necessary info (e.g., score change from `info`).
                * `step_reward = reward_function(theta, obs, act, next_obs, score_change, done)`
                * `total_reward += step_reward`.
            * Store `(trajectory_index, total_reward)`.
        * Sort the stored tuples by `total_reward` in descending order.
        * Return the list of sorted `trajectory_index` values.

### Step 2.5: LLM Ranking

* **Goal:** Rank the *same* sampled trajectories using the LLM based on strategic quality.
* **Input:** List of trajectories, the `rank_trajectories_llm` function (from Step 2.1).
* **Output:** A list of trajectory indices, sorted according to the LLM's preference (most preferred first).
* **Implementation Details:**
    * Call the function developed in Step 2.1:
        `llm_ranked_indices = rank_trajectories_llm(trajectories)`
    * Ensure the output format is consistent (a list of indices).

### Step 2.6: Preference Data Generation

* **Goal:** Create pairs of trajectories $(\tau_i, \tau_j)$ representing preferences, based on disagreements and agreements between $R_{\theta}$ ranking and LLM ranking.
* **Input:** The two rankings: `r_theta_ranked_indices`, `llm_ranked_indices`. The list of `trajectories`.
* **Output:** A preference dataset `D`, typically a list of tuples `(preferred_trajectory, less_preferred_trajectory)`.
* **Implementation Details:** (Following Algorithm 1 in the proposal)
    * Implement a function `generate_preference_data(trajectories, r_theta_ranking, llm_ranking)`:
        * Identify discordant pairs $D_{neg}$: Iterate through all possible pairs of trajectories $(\tau_i, \tau_j)$. If $R_{\theta}$ ranks $\tau_i > \tau_j$ but LLM ranks $\tau_j > \tau_i$, add $(\tau_j, \tau_i)$ to $D_{neg}$ (LLM preference first).
        * Identify concordant pairs $D_{pos}$: Iterate through pairs where both $R_{\theta}$ and LLM agree on the ranking (e.g., both rank $\tau_i > \tau_j$). Sample a number of these pairs, typically equal to $|D_{neg}|$, and add them as $(\tau_i, \tau_j)$ to $D_{pos}$.
        * Combine the datasets: $D = D_{neg} \cup D_{pos}$.
    * Return `D`. The trajectories in `D` can be represented by their actual data or just their indices if the optimization step can access the full trajectory data later.

### Step 2.7: Reward Function Update (Bayesian Optimization)

* **Goal:** Update the reward parameters $\theta$ using the preference data $D$ to better align $R_{\theta}$ with the LLM's preferences.
* **Input:** Preference dataset `D`, current parameters `theta`, the `calculate_parametric_reward` function.
* **Output:** Updated reward parameters `theta_new`.
* **Key Libraries/Functions:** `skopt.gp_minimize` or similar Bayesian Optimization functions. A preference learning model (e.g., Gaussian Process preference learning).
* **Implementation Details:**
    * This step adapts Bayesian Optimization for preference learning. The standard approach involves learning a latent utility function $f(\tau)$ over trajectories. The preference $(\tau_i, \tau_j) \in D$ suggests $f(\tau_i) > f(\tau_j)$. We want to find $\theta$ such that the total reward $R_{\theta}(\tau) = \sum_{t} r_{\theta}(s_t, a_t, s_{t+1})$ correlates well with $f(\tau)$.
    * Define an objective function for the Bayesian Optimizer. This function takes candidate parameters `theta_candidate` and evaluates how well $R_{\theta_{candidate}}$ respects the preferences in `D`.
        * A simple objective could be the negative pairwise accuracy: Iterate through all pairs $(\tau_preferred, \tau_{less\_preferred})$ in `D`. Calculate $R_{\theta_{candidate}}(\tau_{preferred})$ and $R_{\theta_{candidate}}(\tau_{less\_preferred})$. The objective is the fraction of pairs where $R_{\theta_{candidate}}(\tau_{preferred}) \le R_{\theta_{candidate}}(\tau_{less\_preferred})$. The optimizer will minimize this (maximize accuracy).
    * Define the search space for `theta` (bounds for each parameter).
    * Use `skopt.gp_minimize` (or equivalent):
        ```python
        from skopt import gp_minimize
        from skopt.space import Real

        # Define bounds for each parameter in theta
        search_space = [Real(low=-1.0, high=1.0, name=f'theta_{i}') for i in range(len(theta))]

        def objective(theta_candidate):
            accuracy = 0
            for preferred_traj, less_preferred_traj in D:
                 # Calculate total reward for each trajectory using theta_candidate
                 r_pref = sum(calculate_parametric_reward(theta_candidate, *step) for step in preferred_traj)
                 r_less = sum(calculate_parametric_reward(theta_candidate, *step) for step in less_preferred_traj)
                 if r_pref > r_less:
                     accuracy += 1
            # Minimize negative accuracy
            return - (accuracy / len(D))

        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=50, # Number of iterations for Bayesian Optimization
            random_state=42,
            x0=list(theta) # Start search from current theta
        )
        theta_new = np.array(result.x)
        ```
    * Return `theta_new`.

### Step 2.8: Main Training Loop Integration

* **Goal:** Combine all previous steps into the main iterative loop described in Algorithm 1.
* **Input:** Initialized RL agent, initial `theta`, environment, LLM setup, configuration parameters (num trajectories `M`, num RL update steps `k`, total iterations `T`).
* **Output:** A final trained RL agent using the self-aligned reward function, and the final `theta`.
* **Implementation Details:**
    * Initialize `theta` (Step 2.2).
    * Initialize RL agent `model` (Step 1.2).
    * Outer loop `for t = 0 to T-1`:
        * **Inner RL Training:** Train the RL agent `model` using the current `calculate_parametric_reward` with the current `theta` for `k` steps or episodes.
            * This requires modifying the environment wrapper or using SB3 callbacks to inject the parametric reward calculation during `model.learn()`.
        * **Sample Trajectories:** `trajectories = sample_trajectories(model, env, M)` (Step 2.3).
        * **Rank with $R_{\theta}$:** `r_theta_ranking = rank_trajectories_r_theta(trajectories, theta, calculate_parametric_reward)` (Step 2.4).
        * **Rank with LLM:** `llm_ranking = rank_trajectories_llm(trajectories)` (Step 2.5).
        * **Generate Preferences:** `D = generate_preference_data(trajectories, r_theta_ranking, llm_ranking)` (Step 2.6).
        * **Update Theta:** `theta = update_theta_bayesian_opt(D, theta, calculate_parametric_reward, search_space)` (Step 2.7, wrapped in a helper function).
        * Log progress (iteration `t`, current `theta`, size of `D`, Bayesian Opt result).
        * Save intermediate models and `theta` periodically.
    * Return final `model` and `theta`.

---

## Phase 3: Implementing Novel Contributions

**(Corresponds roughly to Weeks 5-7, integrate into Phase 2 loop)**

### Step 3.1: Constrained Reward Editing

* **Goal:** Modify the reward update step (2.7) to only optimize a subset of parameters in $\theta$ or apply constraints.
* **Input:** Specification of which parameters in `theta` are fixed and which are tunable. Modified search space for Bayesian Optimization.
* **Output:** Reward update step respects the constraints.
* **Implementation Details:**
    * Identify the indices of `theta` that should be optimized.
    * Adjust the `search_space` passed to `skopt.gp_minimize` in Step 2.7 to only include dimensions corresponding to the tunable parameters.
    * When calling the `objective` function during optimization, reconstruct the *full* `theta` vector by combining the fixed parameters with the `theta_candidate` values provided by the optimizer for the tunable dimensions.

### Step 3.2: Weighted Preference Ranking

* **Goal:** Modify preference generation (2.6) or reward update (2.7) to account for the strength of preference.
* **Input:** A method to quantify preference strength (e.g., difference in LLM scores/ranks, confidence scores from LLM if available).
* **Output:** Preference data `D` might include weights `w_i` for each pair, or the optimization objective function uses weights.
* **Implementation Details (Option A: Weighted Objective):**
    * Modify the LLM ranking (Step 2.5) to extract not just the order but also confidence scores or rationale that can be mapped to a weight (e.g., LLM says "Trajectory A is significantly better" vs "Trajectory A is slightly better"). This is highly dependent on LLM capabilities and prompt design.
    * Modify the preference data generation (Step 2.6) to store `(preferred_traj, less_preferred_traj, weight)`.
    * Modify the Bayesian Optimization objective function (Step 2.7) to use these weights. For example, instead of `accuracy += 1`, use `accuracy += weight`.
* **Implementation Details (Option B: Modify Pair Selection):**
    * Quantify the *discrepancy* between LLM rank and $R_{\theta}$ rank for discordant pairs.
    * In Step 2.6, prioritize sampling pairs into $D_{neg}$ that have a larger discrepancy, assuming these represent stronger learning signals. The size of $D_{pos}$ could still match the resulting $|D_{neg}|$.

---

## Phase 4: Evaluation

**(Corresponds roughly to Weeks 7-9)**

### Step 4.1: Metrics Implementation

* **Goal:** Implement functions to calculate evaluation metrics from game rollouts.
* **Input:** A trained agent (`model`), the environment (`env`), number of evaluation episodes `N`.
* **Output:** Aggregate metrics (average/max score, average/max tile, win rate).
* **Key Libraries/Functions:** `stable_baselines3.common.evaluation.evaluate_policy` (can be used as a base, but likely need custom logic for specific 2048 metrics).
* **Implementation Details:**
    * Create an evaluation function `evaluate_agent(model, env, num_episodes)`:
        * Loop `num_episodes` times:
            * Run one full episode using `model.predict(obs, deterministic=True)`.
            * Record final score, max tile achieved, whether 2048 tile was reached.
        * Calculate and return average/std dev/max for score and max tile, and the win rate (fraction of episodes reaching 2048).

### Step 4.2: Experiment Execution Script

* **Goal:** Create a script to systematically run training and evaluation for different configurations (baseline, SAILOR, SAILOR+constraints, SAILOR+weighting).
* **Input:** Configuration parameters for each run (which agent type, hyperparameters, number of seeds).
* **Output:** Saved models, logs, and evaluation results for each configuration.
* **Implementation Details:**
    * Use Python's `argparse` to pass configuration options via command line.
    * Structure the code to easily switch between baseline training and the different SAILOR configurations.
    * Loop through different random seeds for robustness.
    * Save results (metrics, final theta) to CSV or JSON files for easy analysis.
    * Leverage AWS (e.g., EC2) for parallel execution of different experiment runs if needed.

### Step 4.3: Results Analysis and Visualization

* **Goal:** Process the saved results and generate plots for the final report.
* **Input:** Saved result files (CSV/JSON).
* **Output:** Learning curves (e.g., average score vs. training iterations/timesteps), comparison bar plots (final performance of different methods), potentially visualizations of learned strategies (e.g., heatmaps of typical end-game board states).
* **Key Libraries/Functions:** `pandas`, `matplotlib.pyplot`, `seaborn`.
* **Implementation Details:**
    * Write scripts to load results using `pandas`.
    * Generate plots using `matplotlib` or `seaborn`.
        * Learning curves: Plot evaluation metrics gathered during training (requires integrating evaluation into the main loop or running it periodically). Use TensorBoard logs if generated.
        * Bar plots: Compare final evaluation metrics across different agent versions (baseline, SAILOR variants) with error bars (std dev across seeds).
    * (Optional) Qualitative analysis: Save some full episode rollouts (sequences of board states) for interesting agents and visualize them.

---

## Phase 5: Stretch Goals (Optional)

**(Corresponds roughly to Weeks 8-10)**

### Step 5.1: Reward Generalizability Test

* **Goal:** Test if the reward function $R_{\theta}$ learned via PPO+SAILOR can benefit a different RL algorithm like DQN.
* **Input:** The final learned `theta`, the `calculate_parametric_reward` function, the environment `env`.
* **Output:** Performance metrics of a DQN agent trained using the learned $R_{\theta}$.
* **Key Libraries/Functions:** `stable_baselines3.DQN`.
* **Implementation Details:**
    * Train a DQN model (`stable_baselines3.DQN`) on the environment.
    * Ensure the environment uses the `calculate_parametric_reward` function with the final `theta` learned by SAILOR (PPO).
    * Evaluate the trained DQN agent using the metrics from Step 4.1.
    * Compare its performance to a DQN trained with the baseline reward function.

### Step 5.2: LLM-Proposed Reward Curricula

* **Goal:** Explore using the LLM to suggest changes to the *structure* or *focus* of the reward function over time, not just parameter values.
* **Input:** LLM, current agent performance/stage.
* **Output:** A sequence of reward function structures or parameter focuses.
* **Implementation Details:** (Highly exploratory)
    * Requires designing prompts for the LLM to analyze the agent's current strategy (e.g., from sampled trajectories) and suggest modifications to the reward features or their weights. E.g., "The agent is merging well but not setting up corners. Suggest modifying the reward parameters `theta` or adding new features to encourage corner play."
    * Implement logic to parse the LLM's suggestions and adapt the `calculate_parametric_reward` function or the `search_space`/constraints in the Bayesian Optimization accordingly. This likely involves significant engineering effort.

---

## AWS & Budget Management

* **Monitor Usage:** Regularly check the AWS Billing console. Set up budget alerts.
* **EC2 Instances:**
    * Use Spot Instances for training/batch jobs where feasible (handle potential interruptions).
    * Choose appropriate instance types (e.g., `g4dn` or `g5` for GPU-accelerated LLM inference/training, `c5` or `m5` for CPU-bound RL rollouts). Size matters for cost.
    * **Shut down instances when not in use!**
* **LLM Costs:**
    * Use smaller models if sufficient (Phi-3-Mini, Mistral-7B).
    * Apply quantization (e.g., GPTQ, AWQ) if using larger models on limited hardware.
    * Optimize inference: use batching, efficient attention mechanisms (Flash Attention if available/compatible).
    * Limit the frequency and scale of LLM ranking steps if budget becomes tight.
* **Data Storage:** Use S3 for storing large datasets (trajectories, model checkpoints) if needed, but be mindful of storage costs and data transfer fees if moving data frequently between services.