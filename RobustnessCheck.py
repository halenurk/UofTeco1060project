import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CORE PARAMETERS (SCALED FOR 40-90 RANGE)
# ==========================================
PARAMS = {
    'N': 100,                
    'rounds': 1000,          # Increased for stable convergence
    'learning_rate': 0.05,   # Lowered for stable learning
    'epsilon': 0.05,         
    
    'theta_L': 1.0,          
    'theta_H': 2.0,          
    'C_H': 0.5,              
    'C_L': 1.0,              
    
    # SCALED SCORING PARAMETERS (10x)
    'alpha': 15.0,           
    'beta': 20.0,            
    'base_boost': 20.0,      # The +20 shift

    'V_A': 5.0,              
    'V_B': 2.0,              
    'V_C': 0.0,              
    
    'q_A': 0.25,             
    'q_B': 0.45,             
}

# Expected baseline scores for the plot text box
H_w = PARAMS['alpha'] * 1 + PARAMS['beta'] * PARAMS['theta_H'] + PARAMS['base_boost']
H_s = PARAMS['alpha'] * 0 + PARAMS['beta'] * PARAMS['theta_H'] + PARAMS['base_boost']
L_w = PARAMS['alpha'] * 1 + PARAMS['beta'] * PARAMS['theta_L'] + PARAMS['base_boost']
L_s = PARAMS['alpha'] * 0 + PARAMS['beta'] * PARAMS['theta_L'] + PARAMS['base_boost']

# ==========================================
# 2. RELATIVE GRADING ENGINE (WITH DELTA INJECTION)
# ==========================================
def run_robustness_simulation(p_high, delta):
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
        
        # Apply the specific delta for this sweep iteration
        noise = np.random.uniform(-delta, delta, size=N)
        scores = PARAMS['alpha'] * efforts + PARAMS['beta'] * thetas + PARAMS['base_boost'] + noise
        
        ranks = np.argsort(np.argsort(scores))
        
        cutoff_A = int(N * (1 - PARAMS['q_A']))
        cutoff_B = int(N * (1 - (PARAMS['q_A'] + PARAMS['q_B'])))
        
        payoffs = np.where(ranks >= cutoff_A, PARAMS['V_A'], 
                  np.where(ranks >= cutoff_B, PARAMS['V_B'], PARAMS['V_C']))
        utilities = payoffs - (efforts * costs)
        
        lr = PARAMS['learning_rate']
        Q_table[np.arange(N), efforts] = (1 - lr) * Q_table[np.arange(N), efforts] + lr * utilities

        # 50-round smoothing for high stability
        if r >= PARAMS['rounds'] - 50:
            history_e_H.append(np.mean(efforts[is_high_skill]) if num_high_skill > 0 else 0)
            history_e_L.append(np.mean(efforts[~is_high_skill]) if num_high_skill < N else 0)

    return np.mean(history_e_H), np.mean(history_e_L)

# ==========================================
# 3. VISUALIZATION FUNCTION
# ==========================================
def plot_robustness_check(data):
    plt.figure(figsize=(11, 6.5))
    
    plt.plot(data['delta_vals'], data['e_H'], marker='o', color='blue', linewidth=2.5, label=r'High-Skill Effort ($e_H$)')
    plt.plot(data['delta_vals'], data['e_L'], marker='s', color='red', linewidth=2.5, label=r'Low-Skill Effort ($e_L$)')
    
    # Draw a vertical line indicating our baseline delta
    plt.axvline(x=10.0, color='gray', linestyle='--', linewidth=2, label=r'Baseline Noise ($\delta = 10$)')
    
    plt.title("Robustness Check: The Impact of Exam Noise on the Rat Race\n(Fixed Demographics: $p=0.50$)", fontsize=15)
    plt.xlabel(r"Exam Noise Amplitude ($\pm \delta$ points)", fontsize=13)
    plt.ylabel("Equilibrium Proportion Working ($e=1$)", fontsize=13)
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add the text box with expected grades
    info_text = (f"Expected Base Scores:\n"
                 f"$H_w$: {H_w:.0f}  |  $H_s$: {H_s:.0f}\n"
                 f"$L_w$: {L_w:.0f}  |  $L_s$: {L_s:.0f}\n"
                 f"Quota: Top {PARAMS['q_A']*100}% get A")
    
    plt.text(0.80, 0.50, info_text, transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='round,pad=0.5'))
             
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.savefig('visual_6_noise_robustness.png', dpi=300)
    print("Generated: visual_6_noise_robustness.png")

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    np.random.seed(1060)
    
    # Sweep delta from 1 point (highly precise) to 30 points (lottery)
    delta_values = np.linspace(1, 30, 30)
    fixed_p = 0.50 
    
    simulation_data = {
        'delta_vals': [],
        'e_H': [], 'e_L': []
    }
    
    print(f"Running robustness sweep at p={fixed_p}...")
    
    for d in delta_values:
        emp_e_H, emp_e_L = run_robustness_simulation(fixed_p, d)
        simulation_data['delta_vals'].append(d)
        simulation_data['e_H'].append(emp_e_H)
        simulation_data['e_L'].append(emp_e_L)

    print("\nGenerating visualization...")
    plot_robustness_check(simulation_data)
    
    print("Exporting data to CSV...")
    with open('visual_6_robustness_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delta', 'High_Skill_Effort', 'Low_Skill_Effort'])
        for d, eh, el in zip(simulation_data['delta_vals'], simulation_data['e_H'], simulation_data['e_L']):
            writer.writerow([round(d, 1), eh, el])
            
    print("All tasks complete!")