import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SIMULATION PARAMETERS
# ==========================================
PARAMS = {
    # Algorithmic Parameters (The Learning Engine)
    'N': 100,                # Number of students (keep at 100 for a clean 10x10 grid)
    'rounds': 100,           # Time horizon for Q-table convergence
    'learning_rate': 0.1,    # How heavily agents weigh recent grades vs past grades
    'epsilon': 0.05,         # Bounded rationality: 5% chance to randomly explore (shirk/work)

 
    'p_high': 0.5,          # Proportion of high-skill students

    # The Classroom (Zubrickas / Spence)
    'theta_L': 1.0,          # Low-skill parameter
    'theta_H': 2.0,          # High-skill parameter
    'C_H': 0.5,              # Single-Crossing: Cost is exactly 1/theta
    'C_L': 1.0,              # Cost of effort for Low-skill # Mine 1.2

    # The Assessment (Perez & Skreta)
    'alpha': 15,            # Effort multiplier
    'beta': 20,             # Skill multiplier
    'delta': 10,            # Uniform noise bound (The "Lottery" factor)
    'base_boost': 20,       # Base score boost to ensure positive scores

    # The Market Wage / Status (Dubey & Geanakoplos / Cotton)
    'V_A': 5.0,              # Convex status premium for top grade # Mine 3.0
    'V_B': 2.0,              # Utility of B
    'V_C': 0.0,              # Utility of C (normalized to zero) # Mine 1.0

    # The Professor's Objective
    'omega_H': 0.9,          # Weight on high-skill effort
    'omega_L': 0.1,          # Weight on low-skill effort
    
    # Relative Grading Quotas (Boleslavsky & Cotton)
    'q_A': 0.25,             # Strict signaling quota
    'q_B': 0.45,             
}


# ==========================================
# 2. THEORETICAL BENCHMARK FUNCTIONS
# ==========================================
def expected_prob(threshold, effort, theta):
    base_score = PARAMS['alpha'] * effort + PARAMS['beta'] * theta + PARAMS['base_boost']
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
# 3. THE CLASSROOM INNER LOOP (ABS ENGINE)
# ==========================================
def run_classroom_simulation(T_A, T_B):
    """Runs N=100 students for 100 rounds and returns final state and empirical W."""
    N = PARAMS['N']
    grid_size = int(np.sqrt(N))
    
    is_high_skill = np.zeros(N, dtype=bool)
    is_high_skill[:int(N * PARAMS['p_high'])] = True
    
    thetas = np.where(is_high_skill, PARAMS['theta_H'], PARAMS['theta_L'])
    costs = np.where(is_high_skill, PARAMS['C_H'], PARAMS['C_L'])
    
    Q_table = np.ones((N, 2)) * PARAMS['V_A']
    efforts = np.zeros(N, dtype=int)
    
    for _ in range(PARAMS['rounds']):
        # Action Selection
        explore = np.random.rand(N) < PARAMS['epsilon']
        efforts = np.where(explore, np.random.choice([0, 1], size=N), np.argmax(Q_table, axis=1))
        
        # Realize Scores and Utilities
        noise = np.random.uniform(-PARAMS['delta'], PARAMS['delta'], size=N)
        scores = PARAMS['alpha'] * efforts + PARAMS['beta'] * thetas + noise + PARAMS['base_boost']
        
        payoffs = np.where(scores >= T_A, PARAMS['V_A'], 
                  np.where(scores >= T_B, PARAMS['V_B'], PARAMS['V_C']))
        utilities = payoffs - (efforts * costs)
        
        # Update Q-Table
        lr = PARAMS['learning_rate']
        Q_table[np.arange(N), efforts] = (1 - lr) * Q_table[np.arange(N), efforts] + lr * utilities

    # 1. Calculate the "pure" strategy directly from the brain (ignores epsilon noise)
    pure_strategy = np.argmax(Q_table, axis=1)
    
    # 2. Calculate Professor's Utility (W) based on this pure strategy
    mean_e_H = np.mean(pure_strategy[is_high_skill])
    mean_e_L = np.mean(pure_strategy[~is_high_skill])
    pure_W = PARAMS['omega_H'] * mean_e_H + PARAMS['omega_L'] * mean_e_L
    
    # 3. Return the pure grid and the perfectly matched pure Utility
    return pure_strategy.reshape((grid_size, grid_size)), pure_W

# ==========================================
# 4. THE PROFESSOR'S OUTER LOOP & VISUALIZATION
# ==========================================
def generate_phase_space_map():
    # 1. TRIMMED AXES (Cut off the completely empty row and column)
    T_A_vals = np.arange(90.0, 40.0, -5.0) # Stops at 45 (removes the empty T_A=40 row)
    T_B_vals = np.arange(40.0, 90.0, 5.0)  # Stops at 85 (removes the empty T_B=90 column)
    
    fig, axes = plt.subplots(len(T_A_vals), len(T_B_vals), figsize=(16, 16))
    fig.patch.set_facecolor('#f4f4f4')
    
    cmap = ListedColormap(['#ff4c4c', '#4cff4c']) # Red=0, Green=1
    
    results = {}
    max_W = -1.0
    optimal_coords = [] # CHANGED: Now a list to hold multiple winners

    print("Professor is mapping the policy space...")
    
    # Run the nested simulations
    for i, t_a in enumerate(T_A_vals):
        for j, t_b in enumerate(T_B_vals):
            ax = axes[i, j]
            
            # Constraint: A threshold must be strictly higher than B threshold
            if t_b >= t_a:
                ax.axis('off') 
                continue
                
            # Run simulation
            final_grid, emp_W = run_classroom_simulation(t_a, t_b)
            results[(i, j)] = emp_W
            
            # MULTIPLE OPTIMAL HIGHLIGHT LOGIC:
            # We round to 2 decimal places so that any cell visually displaying
            # the maximum value gets a star, preventing invisible micro-decimal ties.
            if round(emp_W, 2) > round(max_W, 2):
                max_W = emp_W
                optimal_coords = [(i, j)] # Clear list, set new absolute max
            elif round(emp_W, 2) == round(max_W, 2):
                optimal_coords.append((i, j)) # Tie found, add to list!
                
            # Plot the grid
            ax.imshow(final_grid, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Label axes on the outer edges
            if j == 0 or (j > 0 and t_b >= T_A_vals[i-1] if i > 0 else False): 
                ax.set_ylabel(f"$T_A={int(t_a)}$", fontsize=12, fontweight='bold')
            if i == len(T_A_vals)-1 or t_b >= t_a - 5.0:
                ax.set_xlabel(f"$T_B={int(t_b)}$", fontsize=12, fontweight='bold')
            
            # Overlay Utility text
            color = 'black' if emp_W > 0.5 else 'white' 
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            ax.text(4.5, 4.5, f"{emp_W:.2f}", color='black', ha='center', va='center', 
                    fontsize=10, fontweight='bold', bbox=bbox_props)

    # Highlight ALL empirical optimal policies
    if optimal_coords:
        for (opt_i, opt_j) in optimal_coords:
            for spine in axes[opt_i, opt_j].spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(4)
                spine.set_zorder(10)
            axes[opt_i, opt_j].set_title("★ OPTIMAL ★", color='goldenrod', fontsize=11, fontweight='bold', pad=4)
    
    # Maximize the grid
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    
    # 2. LOWERED INFORMATION HUB
    info_text = (
        "$\\bf{Equilibrium\\ Phase\\ Space\\ Map}$\n"
        "Empirical Search for Optimal Policy\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "$\\bf{Grid\\ Cell\\ Guide:}$\n"
        " • Top half of cell = High-Skill Students\n"
        " • Bottom half of cell = Low-Skill Students\n"
        " • Center Text = Professor's Utility ($W$)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"$\\bf{{Payoffs:}}$ $V_A={PARAMS.get('V_A', 3.0)}$, $V_B={PARAMS.get('V_B', 2.0)}$, $V_C={PARAMS.get('V_C', 1.0)}$\n"
        f"$\\bf{{Costs:}}$ $C_H={PARAMS.get('C_H', 0.5)}$, $C_L={PARAMS.get('C_L', 1.2)}$\n"
        f"$\\bf{{Skill (\\theta):}}$ $\\theta_H={PARAMS.get('theta_H', 2.0)}$, $\\theta_L={PARAMS.get('theta_L', 1.0)}$\n"
        f"$\\bf{{Production:}}$ $S = {PARAMS.get('alpha', 15.0)}e + {PARAMS.get('beta', 20.0)}\\theta + {PARAMS.get('base_boost', 20.0)} \\pm {PARAMS.get('delta', 10.0)}$\n"
        f"$\\bf{{Prof\\ Weights:}}$ $\\omega_H={PARAMS.get('omega_H', 0.9)}$, $\\omega_L={PARAMS.get('omega_L', 0.1)}$"
    )

    # Pushed the Y-coordinate down from 0.50 to 0.42
    fig.text(0.75, 0.20, info_text, fontsize=15, va='center', ha='center',
             bbox=dict(facecolor='white', edgecolor='#ced4da', boxstyle='round,pad=1', alpha=0.95))

    # 3. SINGLE-LINE LEGEND
    import matplotlib.patches as mpatches
    work_patch = mpatches.Patch(color='#4cff4c', label='Work')
    shirk_patch = mpatches.Patch(color='#ff4c4c', label='Shirk')
    
    # Set ncol=2 for a single row, and pushed the Y-coordinate down to 0.24
    fig.legend(handles=[work_patch, shirk_patch], loc='center', ncol=2, fontsize=15, bbox_to_anchor=(0.75, 0.05))
    
    plt.savefig('visual_phase_space_map_fullscreen.jpg', dpi=300)
    print("Fullscreen Phase Space Map generated successfully!")
    # plt.show() # Uncomment if you want to view it interactively

if __name__ == "__main__":
    # Lock the randomness for perfect reproducibility
    np.random.seed(2026)
    generate_phase_space_map()