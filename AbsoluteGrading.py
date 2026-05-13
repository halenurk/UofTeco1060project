import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE PARAMETERS (SCALED FOR 40-90 RANGE)
# ==========================================
PARAMS = {
    # Algorithmic Parameters
    'N': 100,                
    'rounds': 1000,          # Increased for stable convergence
    'learning_rate': 0.05,   # Lowered to prevent overreacting to noise
    'epsilon': 0.05,         
    
    # Economic Parameters (Costs & Types)
    'theta_L': 1.0,          
    'theta_H': 2.0,          
    'C_H': 0.5,              
    'C_L': 1.0,              
    
    # Scaled Scoring Parameters (10x + 20)
    'alpha': 15.0,           
    'beta': 20.0,            
    'base_boost': 20.0,      
    'delta': 10.0,           # Noise is now +/- 10 points
    
    # Convex Status Payoffs
    'V_A': 5.0,              
    'V_B': 2.0,              
    'V_C': 0.0,              
    
    # Professor Objective Weights
    'omega_H': 0.9,          
    'omega_L': 0.1,          
}

# Calculate expected base scores for the plot text box
H_w = PARAMS['alpha'] * 1 + PARAMS['beta'] * PARAMS['theta_H'] + PARAMS['base_boost']
H_s = PARAMS['alpha'] * 0 + PARAMS['beta'] * PARAMS['theta_H'] + PARAMS['base_boost']
L_w = PARAMS['alpha'] * 1 + PARAMS['beta'] * PARAMS['theta_L'] + PARAMS['base_boost']
L_s = PARAMS['alpha'] * 0 + PARAMS['beta'] * PARAMS['theta_L'] + PARAMS['base_boost']

# ==========================================
# 2. ENGINES
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

def run_classroom_simulation(T_A, T_B, p_high):
    N = PARAMS['N']
    is_high_skill = np.zeros(N, dtype=bool)
    num_high_skill = int(round(N * p_high))
    is_high_skill[:num_high_skill] = True
    
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
        # Apply the scaled score equation
        scores = PARAMS['alpha'] * efforts + PARAMS['beta'] * thetas + PARAMS['base_boost'] + noise
        
        payoffs = np.where(scores >= T_A, PARAMS['V_A'], 
                  np.where(scores >= T_B, PARAMS['V_B'], PARAMS['V_C']))
        utilities = payoffs - (efforts * costs)
        
        lr = PARAMS['learning_rate']
        Q_table[np.arange(N), efforts] = (1 - lr) * Q_table[np.arange(N), efforts] + lr * utilities

        # Smooth over final 50 rounds
        if r >= PARAMS['rounds'] - 50:
            history_e_H.append(np.mean(efforts[is_high_skill]) if num_high_skill > 0 else 0)
            history_e_L.append(np.mean(efforts[~is_high_skill]) if num_high_skill < N else 0)

    mean_e_H = np.mean(history_e_H)
    mean_e_L = np.mean(history_e_L)
    empirical_W = PARAMS['omega_H'] * mean_e_H + PARAMS['omega_L'] * mean_e_L
    
    return empirical_W, efforts.reshape((10, 10))

# ==========================================
# 3. OPTIMIZER
# ==========================================
def find_optimal_policy(p_high, mode='empirical'):
    # Search grid is now scaled to standard grading metrics (40 to 90)
    T_A_vals = np.arange(90.0, 35.0, -5.0)
    T_B_vals = np.arange(40.0, 95.0, 5.0)
    
    max_W = -np.inf
    opt_T_A, opt_T_B = None, None
    best_grid = None
    
    for t_a in T_A_vals:
        for t_b in T_B_vals:
            if t_b >= t_a: continue
                
            if mode == 'empirical':
                W, final_grid = run_classroom_simulation(t_a, t_b, p_high)
            else:
                W = get_theoretical_W(t_a, t_b)
                final_grid = None
            
            if W > max_W + 1e-6:
                max_W = W
                opt_T_A, opt_T_B = t_a, t_b
                best_grid = final_grid
            elif abs(W - max_W) < 1e-6:
                if opt_T_A is None or t_a < opt_T_A:
                    opt_T_A, opt_T_B = t_a, t_b
                    best_grid = final_grid
                elif t_a == opt_T_A and t_b < opt_T_B:
                    opt_T_A, opt_T_B = t_a, t_b
                    best_grid = final_grid
                    
    return opt_T_A, opt_T_B, best_grid

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================
def plot_overlaid_statics(data):
    plt.figure(figsize=(12, 7.5))
    
    plt.plot(data['p_values'], data['theo_ta'], marker='o', linestyle='--', color='darkblue', 
             markerfacecolor='none', linewidth=2, markersize=8, label=r'Theoretical $T_A^*$')
    plt.plot(data['p_values'], data['theo_tb'], marker='s', linestyle='--', color='darkorange', 
             markerfacecolor='none', linewidth=2, markersize=8, label=r'Theoretical $T_B^*$')
             
    plt.plot(data['p_values'], data['emp_ta'], marker='o', color='blue', linewidth=2.5, label=r'Empirical $T_A^*$ (ABS)')
    plt.plot(data['p_values'], data['emp_tb'], marker='s', color='orange', linewidth=2.5, label=r'Empirical $T_B^*$ (ABS)')
    
    plt.title("The Principal's Reaction Function:\nAdapting Fixed Grading Standards to Demographics", fontsize=15)
    plt.xlabel("Proportion of High-Skill Students ($p$)", fontsize=13)
    plt.ylabel("Grading Threshold (Raw Score)", fontsize=13)
    plt.xticks(np.arange(0.05, 1.0, 0.05), rotation=45)
    plt.yticks(np.arange(40, 95, 5))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Injection of Parameter Text Box
    info_text = (f"Simulation Parameters:\n"
                 f"$H_w$: {H_w:.0f} | $H_s$: {H_s:.0f}\n"
                 f"$L_w$: {L_w:.0f} | $L_s$: {L_s:.0f}\n"
                 f"Noise ($\delta$): $\pm${PARAMS['delta']}\n"
                 f"Costs ($C_H, C_L$): {PARAMS['C_H']}, {PARAMS['C_L']}\n"
                 f"Payoffs ($V_A, V_B, V_C$): {PARAMS['V_A']}, {PARAMS['V_B']}, {PARAMS['V_C']}")
    
    plt.text(0.78, 0.20, info_text, transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='round,pad=0.5'))
             
    plt.legend(loc='lower left', fontsize=11, ncol=2)
    plt.tight_layout()
    plt.savefig('visual_4a_overlaid_statics.png', dpi=300)

def plot_equilibrium_filmstrip(data):
    target_ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig, axes = plt.subplots(1, 9, figsize=(18, 4))
    fig.patch.set_facecolor('#f4f4f4')
    cmap = ListedColormap(['#ff4c4c', '#4cff4c'])
    
    for idx, target_p in enumerate(target_ps):
        p_key = round(target_p, 2)
        t_a, t_b, final_grid = data['emp_grids'][p_key]
        
        ax = axes[idx]
        ax.imshow(final_grid, cmap=cmap, vmin=0, vmax=1)
        
        boundary_row = target_p * 10 - 0.5 
        ax.axhline(y=boundary_row, color='black', linewidth=3, linestyle='-')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$p = {target_p:.1f}$", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(f"$T_A^* = {t_a:.0f}$\n$T_B^* = {t_b:.0f}$", fontsize=12, labelpad=8)

    plt.suptitle("Classroom Equilibria Under Empirically Optimal Absolute Grading\n" +
                 "(Black line separates High-Skill on top from Low-Skill on bottom)", 
                 fontsize=16, fontweight='bold', y=1.1)
                 
    import matplotlib.patches as mpatches
    work_patch = mpatches.Patch(color='#4cff4c', label='Work Equilibrium')
    shirk_patch = mpatches.Patch(color='#ff4c4c', label='Shirk Equilibrium')
    fig.legend(handles=[work_patch, shirk_patch], loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig('visual_4b_equilibrium_filmstrip.png', dpi=300, bbox_inches='tight')

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    np.random.seed(1060) # Lock the randomness
    p_values = np.linspace(0.05, 0.95, 19)
    
    simulation_data = {
        'p_values': [],
        'theo_ta': [], 'theo_tb': [],
        'emp_ta': [], 'emp_tb': [],
        'emp_grids': {} 
    }
    
    print("Running unified demographic sweep (this will take a moment)...")
    
    for p in p_values:
        print(f"Optimizing for p = {p:.2f}...")
        t_ta, t_tb, _ = find_optimal_policy(p, mode='theoretical')
        e_ta, e_tb, e_grid = find_optimal_policy(p, mode='empirical')
        
        simulation_data['p_values'].append(p)
        simulation_data['theo_ta'].append(t_ta)
        simulation_data['theo_tb'].append(t_tb)
        simulation_data['emp_ta'].append(e_ta)
        simulation_data['emp_tb'].append(e_tb)
        
        p_key = round(p, 2)
        simulation_data['emp_grids'][p_key] = (e_ta, e_tb, e_grid)

    print("\nData collected. Generating visualizations...")
    plot_overlaid_statics(simulation_data)
    plot_equilibrium_filmstrip(simulation_data)
    
    print("Exporting data to CSV...")
    with open('visual_4_absolute_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['p_value', 'Theo_TA', 'Theo_TB', 'Emp_TA', 'Emp_TB'])
        for i in range(len(simulation_data['p_values'])):
            writer.writerow([
                round(simulation_data['p_values'][i], 2),
                simulation_data['theo_ta'][i],
                simulation_data['theo_tb'][i],
                simulation_data['emp_ta'][i],
                simulation_data['emp_tb'][i]
            ])
            
    print("All tasks complete! Check your folder for visual_4_absolute_data.csv")