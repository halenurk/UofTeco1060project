import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE PARAMETERS
# ==========================================
PARAMS = {
    # Algorithmic Parameters (The Learning Engine)
    'N': 100,                # Number of students (keep at 100 for a clean 10x10 grid)
    'rounds': 100,           # Time horizon for Q-table convergence
    'learning_rate': 0.1,    # How heavily agents weigh recent grades vs past grades
    'epsilon': 0.05,         # Bounded rationality: 5% chance to randomly explore (shirk/work)

    # The Classroom (Zubrickas / Spence)
    'theta_L': 1.0,          # Low-skill parameter
    'theta_H': 2.0,          # High-skill parameter
    'C_H': 0.5,              # Single-Crossing: Cost is exactly 1/theta
    'C_L': 1.0,              # Cost of effort for Low-skill

    # The Assessment (Perez & Skreta)
    'alpha': 1.5,            # Effort multiplier
    'beta': 2.0,             # Skill multiplier
    'delta': 1.0,            # Uniform noise bound (The "Lottery" factor)

    # The Market Wage / Status (Dubey & Geanakoplos / Cotton)
    'V_A': 5.0,              # Convex status premium for top grade
    'V_B': 2.0,              # Utility of B
    'V_C': 0.0,              # Utility of C (normalized to zero)

    # The Professor's Objective
    'omega_H': 0.9,          # Weight on high-skill effort
    'omega_L': 0.1,          # Weight on low-skill effort
    
    # Relative Grading Quotas (Boleslavsky & Cotton)
    'q_A': 0.25,             # Strict signaling quota
    'q_B': 0.45,             
}

# ==========================================
# 2. THEORETICAL ENGINE (ANALYTICAL BENCHMARK)
# ==========================================
def expected_prob(threshold, effort, theta):
    base_score = PARAMS['alpha'] * effort + PARAMS['beta'] * theta
    target_noise = threshold - base_score
    if target_noise <= -PARAMS['delta']: return 1.0
    elif target_noise >= PARAMS['delta']: return 0.0
    else: return (PARAMS['delta'] - target_noise) / (2 * PARAMS['delta'])

def theoretical_best_response(theta, cost_val, T_A, T_B):
    EU = {}
    for e in [0, 1]:
        cost = cost_val if e == 1 else 0
        p_A = expected_prob(T_A, e, theta)
        p_B = expected_prob(T_B, e, theta)
        expected_u = PARAMS['V_C'] + (PARAMS['V_A'] - PARAMS['V_B']) * p_A + \
                     (PARAMS['V_B'] - PARAMS['V_C']) * p_B - cost
        EU[e] = expected_u
    return 1 if EU[1] > EU[0] else 0

def get_theoretical_W(T_A, T_B):
    e_H_star = theoretical_best_response(PARAMS['theta_H'], PARAMS['C_H'], T_A, T_B)
    e_L_star = theoretical_best_response(PARAMS['theta_L'], PARAMS['C_L'], T_A, T_B)
    return PARAMS['omega_H'] * e_H_star + PARAMS['omega_L'] * e_L_star

# ==========================================
# 3. EMPIRICAL ENGINE (BOUNDED RATIONALITY)
# ==========================================
def run_classroom_simulation(T_A, T_B, p_high):
    """Runs N=100 students for 100 rounds with dynamic demographics and smoothed output."""
    N = PARAMS['N']
    is_high_skill = np.zeros(N, dtype=bool)
    is_high_skill[:int(N * p_high)] = True
    
    thetas = np.where(is_high_skill, PARAMS['theta_H'], PARAMS['theta_L'])
    costs = np.where(is_high_skill, PARAMS['C_H'], PARAMS['C_L'])
    
    Q_table = np.ones((N, 2)) * PARAMS['V_A']
    efforts = np.zeros(N, dtype=int)
    
    history_e_H = []
    history_e_L = []
    
    for r in range(PARAMS['rounds']):
        explore = np.random.rand(N) < PARAMS['epsilon']
        efforts = np.where(explore, np.random.choice([0, 1], size=N), np.argmax(Q_table, axis=1))
        
        noise = np.random.uniform(-PARAMS['delta'], PARAMS['delta'], size=N)
        scores = PARAMS['alpha'] * efforts + PARAMS['beta'] * thetas + noise
        
        payoffs = np.where(scores >= T_A, PARAMS['V_A'], 
                  np.where(scores >= T_B, PARAMS['V_B'], PARAMS['V_C']))
        utilities = payoffs - (efforts * costs)
        
        lr = PARAMS['learning_rate']
        Q_table[np.arange(N), efforts] = (1 - lr) * Q_table[np.arange(N), efforts] + lr * utilities

        # Smooth over the final 20 rounds to eliminate exploration noise
        if r >= PARAMS['rounds'] - 20:
            history_e_H.append(np.mean(efforts[is_high_skill]) if p_high > 0 else 0)
            history_e_L.append(np.mean(efforts[~is_high_skill]) if p_high < 1 else 0)

    mean_e_H = np.mean(history_e_H)
    mean_e_L = np.mean(history_e_L)
    
    return PARAMS['omega_H'] * mean_e_H + PARAMS['omega_L'] * mean_e_L

# ==========================================
# 4. OPTIMIZER (WITH TIE-BREAKING)
# ==========================================
def find_optimal_policy(p_high, mode='empirical'):
    """Scans the phase space. Mode can be 'empirical' or 'theoretical'."""
    T_A_vals = np.arange(7.0, 1.5, -0.5)
    T_B_vals = np.arange(2.0, 7.5, 0.5)
    
    max_W = -np.inf
    opt_T_A, opt_T_B = None, None
    
    for t_a in T_A_vals:
        for t_b in T_B_vals:
            if t_b >= t_a: continue
                
            if mode == 'empirical':
                W = run_classroom_simulation(t_a, t_b, p_high)
            else:
                W = get_theoretical_W(t_a, t_b)
            
            # Tie-breaking logic: If W is equal (within float tolerance), prefer lower grading thresholds
            if W > max_W + 1e-6:
                max_W = W
                opt_T_A, opt_T_B = t_a, t_b
            elif abs(W - max_W) < 1e-6:
                if opt_T_A is None or t_a < opt_T_A:
                    opt_T_A, opt_T_B = t_a, t_b
                elif t_a == opt_T_A and t_b < opt_T_B:
                    opt_T_A, opt_T_B = t_a, t_b
                    
    return opt_T_A, opt_T_B, max_W

# ==========================================
# 5. SWEEP & VISUALIZATION
# ==========================================
def plot_overlaid_statics():
    # Sweep p from 0.05 to 0.95 in 19 highly granular steps
    p_values = np.linspace(0.05, 0.95, 19) 
    
    emp_T_A_list, emp_T_B_list = [], []
    theo_T_A_list, theo_T_B_list = [], []
    
    print("Running theoretical and empirical sweeps across demographics...")
    
    for p in p_values:
        print(f"Optimizing for p = {p:.2f}...")
        
        # Empirical mapping
        e_ta, e_tb, _ = find_optimal_policy(p, mode='empirical')
        emp_T_A_list.append(e_ta)
        emp_T_B_list.append(e_tb)
        
        # Theoretical mapping
        t_ta, t_tb, _ = find_optimal_policy(p, mode='theoretical')
        theo_T_A_list.append(t_ta)
        theo_T_B_list.append(t_tb)
        
    plt.figure(figsize=(12, 7))
    
    # Plot Theoretical Benchmark (Dashed lines, open markers)
    plt.plot(p_values, theo_T_A_list, marker='o', linestyle='--', color='darkblue', markerfacecolor='none', 
             linewidth=2, markersize=8, label=r'Theoretical $T_A^*$')
    plt.plot(p_values, theo_T_B_list, marker='s', linestyle='--', color='darkorange', markerfacecolor='none', 
             linewidth=2, markersize=8, label=r'Theoretical $T_B^*$')
             
    # Plot Empirical Simulation (Solid lines, filled markers)
    plt.plot(p_values, emp_T_A_list, marker='o', color='blue', linewidth=2.5, label=r'Empirical $T_A^*$ (ABS)')
    plt.plot(p_values, emp_T_B_list, marker='s', color='orange', linewidth=2.5, label=r'Empirical $T_B^*$ (ABS)')
    
    plt.title("The Principal's Reaction Function:\nEmpirical Simulation vs. Theoretical Benchmark", fontsize=15)
    plt.xlabel("Proportion of High-Skill Students ($p$)", fontsize=13)
    plt.ylabel("Grading Threshold (Raw Score)", fontsize=13)
    plt.xticks(np.arange(0.05, 1.0, 0.05), rotation=45)
    plt.yticks(np.arange(2.0, 7.5, 0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Organize legend into two columns
    plt.legend(loc='lower left', fontsize=11, ncol=2)
    
    plt.tight_layout()
    plt.savefig('visual_4_overlaid_statics.png', dpi=300)
    print("Overlaid comparative statics generated successfully!")
    # plt.show()

if __name__ == "__main__":
    plot_overlaid_statics()
#if __name__ == "__main__":
#    plot_comparative_statics()