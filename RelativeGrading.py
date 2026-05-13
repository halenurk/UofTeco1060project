import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE PARAMETERS (RELATIVE GRADING)
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
# 2. RELATIVE GRADING ENGINE
# ==========================================
def run_relative_simulation(p_high):
    """Runs the ABS and returns smoothed efforts and the final 10x10 grid."""
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
        scores = PARAMS['alpha'] * efforts + PARAMS['beta'] * thetas + noise
        
        # Rank scores: Double argsort gives the rank (0 is lowest, N-1 is highest)
        ranks = np.argsort(np.argsort(scores))
        
        cutoff_A = int(N * (1 - PARAMS['q_A']))
        cutoff_B = int(N * (1 - (PARAMS['q_A'] + PARAMS['q_B'])))
        
        payoffs = np.where(ranks >= cutoff_A, PARAMS['V_A'], 
                  np.where(ranks >= cutoff_B, PARAMS['V_B'], PARAMS['V_C']))
        utilities = payoffs - (efforts * costs)
        
        lr = PARAMS['learning_rate']
        Q_table[np.arange(N), efforts] = (1 - lr) * Q_table[np.arange(N), efforts] + lr * utilities

        # 20-round smoothing for the line graph metrics
        if r >= PARAMS['rounds'] - 20:
            history_e_H.append(np.mean(efforts[is_high_skill]) if num_high_skill > 0 else 0)
            history_e_L.append(np.mean(efforts[~is_high_skill]) if num_high_skill < N else 0)

    mean_e_H = np.mean(history_e_H)
    mean_e_L = np.mean(history_e_L)
    
    return mean_e_H, mean_e_L, efforts.reshape((10, 10))

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_relative_tipping_point(data):
    plt.figure(figsize=(10, 6))
    
    plt.plot(data['p_values'], data['e_H'], marker='o', color='blue', linewidth=2.5, label=r'High-Skill ($\theta_H$) Effort')
    plt.plot(data['p_values'], data['e_L'], marker='s', color='red', linewidth=2.5, label=r'Low-Skill ($\theta_L$) Effort')
    
    plt.axvline(x=PARAMS['q_A'], color='gold', linestyle='--', linewidth=2.5, 
                label=f"A-Grade Quota ($q_A = {PARAMS['q_A']}$)")
    
    plt.title("The Demographic Tipping Point (Relative Grading):\nStrategic Externalities and the Elite Rat Race", fontsize=14)
    plt.xlabel("Proportion of High-Skill Students ($p$)", fontsize=12)
    plt.ylabel("Equilibrium Proportion Working ($e=1$)", fontsize=12)
    plt.xticks(np.arange(0.05, 1.0, 0.05), rotation=45)
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visual_5a_relative_tipping_point.png', dpi=300)
    print("Generated: visual_5a_relative_tipping_point.png")

def plot_relative_filmstrip(data):
    # Sample 9 specific demographic points for the filmstrip
    target_ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    fig, axes = plt.subplots(1, 9, figsize=(18, 4))
    fig.patch.set_facecolor('#f4f4f4')
    cmap = ListedColormap(['#ff4c4c', '#4cff4c']) # Red=0, Green=1
    
    for idx, target_p in enumerate(target_ps):
        p_key = round(target_p, 2)
        final_grid = data['grids'][p_key]
        
        ax = axes[idx]
        ax.imshow(final_grid, cmap=cmap, vmin=0, vmax=1)
        
        boundary_row = target_p * 10 - 0.5 
        ax.axhline(y=boundary_row, color='black', linewidth=3, linestyle='-')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$p = {target_p:.1f}$", fontsize=14, fontweight='bold', pad=10)
        
        # Add descriptive text below highlighting the regime phase
        regime_text = "Slack Phase" if target_p < PARAMS['q_A'] else "Rat Race"
        color = 'red' if target_p < PARAMS['q_A'] else 'green'
        ax.set_xlabel(f"Regime:\n{regime_text}", fontsize=12, fontweight='bold', color=color, labelpad=8)

    plt.suptitle(f"Classroom Equilibria Under Relative Grading (A-Quota = {PARAMS['q_A']})\n" +
                 "(Black line separates High-Skill on top from Low-Skill on bottom)", 
                 fontsize=16, fontweight='bold', y=1.1)
                 
    import matplotlib.patches as mpatches
    work_patch = mpatches.Patch(color='#4cff4c', label='Work Equilibrium')
    shirk_patch = mpatches.Patch(color='#ff4c4c', label='Shirk Equilibrium')
    fig.legend(handles=[work_patch, shirk_patch], loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig('visual_5b_relative_filmstrip.png', dpi=300, bbox_inches='tight')
    print("Generated: visual_5b_relative_filmstrip.png")

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Set the random seed for perfect reproducibility
    np.random.seed(1060) # Named after the course!
    
    p_values = np.linspace(0.05, 0.95, 19)
    
    simulation_data = {
        'p_values': [],
        'e_H': [], 'e_L': [],
        'grids': {} 
    }
    
    print("Running relative grading demographic sweep...")
    
    for p in p_values:
        emp_e_H, emp_e_L, final_grid = run_relative_simulation(p)
        
        simulation_data['p_values'].append(p)
        simulation_data['e_H'].append(emp_e_H)
        simulation_data['e_L'].append(emp_e_L)
        
        p_key = round(p, 2)
        simulation_data['grids'][p_key] = final_grid

    print("\nData collected. Generating visualizations...")
    plot_relative_tipping_point(simulation_data)
    plot_relative_filmstrip(simulation_data)
    
    # Export the raw data to a CSV file
    print("Exporting data to CSV...")
    with open('visual_5_relative_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['p_value', 'High_Skill_Effort', 'Low_Skill_Effort'])
        for p, eh, el in zip(simulation_data['p_values'], simulation_data['e_H'], simulation_data['e_L']):
            writer.writerow([round(p, 2), eh, el])
            
    print("All tasks complete! Check your folder for visual_5_relative_data.csv")