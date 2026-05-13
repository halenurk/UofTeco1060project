import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import warnings

# Suppress warnings for cleaner output during animation generation
warnings.filterwarnings('ignore')

# ==========================================
# 1. SIMULATION PARAMETERS
# ==========================================
PARAMS = {
    # Algorithmic Parameters (The Learning Engine)
    'N': 400,                # Number of students (perfect square for a clean grid)
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

    'p_high': 0.5           # Proportion of high-skill students in the population
}

# ==========================================
# 2. THEORETICAL & UTILITY FUNCTIONS
# ==========================================
def calculate_grade_and_utility(effort, theta, cost_val):
    """Calculates realized raw score, grade payoff, and final utility."""
    # S = alpha*e + beta*theta + epsilon
    noise = np.random.uniform(-PARAMS['delta'], PARAMS['delta'])
    S = PARAMS['alpha'] * effort + PARAMS['beta'] * theta + noise
    
    if S >= PARAMS['T_A']:
        payoff = PARAMS['V_A']
    elif S >= PARAMS['T_B']:
        payoff = PARAMS['V_B']
    else:
        payoff = PARAMS['V_C']
        
    cost = cost_val if effort == 1 else 0
    return payoff - cost

def expected_prob(threshold, effort, theta):
    """Analytic expected probability of crossing a threshold given uniform noise."""
    base_score = PARAMS['alpha'] * effort + PARAMS['beta'] * theta
    # P(S >= T) = P(noise >= T - base_score)
    target_noise = threshold - base_score
    
    # CDF of Uniform[-delta, delta]
    if target_noise <= -PARAMS['delta']:
        return 1.0
    elif target_noise >= PARAMS['delta']:
        return 0.0
    else:
        return (PARAMS['delta'] - target_noise) / (2 * PARAMS['delta'])

def theoretical_best_response(theta, cost_val, T_A, T_B):
    """Calculates if working (e=1) yields higher expected utility than shirking (e=0)."""
    EU = {}
    for e in [0, 1]:
        cost = cost_val if e == 1 else 0
        p_A = expected_prob(T_A, e, theta)
        p_B = expected_prob(T_B, e, theta)
        # Probability of exactly B is P(S >= T_B) - P(S >= T_A)
        # Utility formula: V_C + (V_A - V_B)P(A) + (V_B - V_C)P(B) - C
        expected_u = PARAMS['V_C'] + (PARAMS['V_A'] - PARAMS['V_B']) * p_A + \
                     (PARAMS['V_B'] - PARAMS['V_C']) * p_B - cost
        EU[e] = expected_u
    return 1 if EU[1] > EU[0] else 0

# ==========================================
# 3. AGENT-BASED SIMULATION ENGINE
# ==========================================
class ClassroomABS:
    def __init__(self):
        self.N = PARAMS['N']
        self.grid_size = int(np.sqrt(self.N))
        
        # Initialize Types (Top half High-skill, Bottom half Low-skill for grid visualization)
        self.is_high_skill = np.zeros(self.N, dtype=bool)
        self.is_high_skill[:int(self.N * PARAMS['p_high'])] = True
        
        self.thetas = np.where(self.is_high_skill, PARAMS['theta_H'], PARAMS['theta_L'])
        self.costs = np.where(self.is_high_skill, PARAMS['C_H'], PARAMS['C_L'])
        
        # Rolling average utility tracker: Q[agent, action]
        # Initialize optimistically to encourage early exploration
        self.Q_table = np.ones((self.N, 2)) * PARAMS['V_A'] 
        
        # History tracking
        self.effort_history = []
        self.grid_history = []

    def step(self):
        # 1. Action Selection (epsilon-greedy based on rolling average)
        efforts = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if np.random.rand() < PARAMS['epsilon']:
                efforts[i] = np.random.choice([0, 1]) # Explore
            else:
                efforts[i] = np.argmax(self.Q_table[i]) # Exploit
                
        # 2. Assessment and Utilities
        utilities = np.zeros(self.N)
        for i in range(self.N):
            u = calculate_grade_and_utility(efforts[i], self.thetas[i], self.costs[i])
            utilities[i] = u
            
            # 3. Update Rolling Average (Best-Response Learning)
            lr = PARAMS['learning_rate']
            self.Q_table[i, efforts[i]] = (1 - lr) * self.Q_table[i, efforts[i]] + lr * u
            
        # Track metrics
        self.effort_history.append({
            'high_work_pct': np.mean(efforts[self.is_high_skill]),
            'low_work_pct': np.mean(efforts[~self.is_high_skill])
        })
        self.grid_history.append(efforts.reshape((self.grid_size, self.grid_size)))

    def run(self):
        for _ in range(PARAMS['rounds']):
            self.step()

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================
def plot_convergence(history):
    """Visual 1: Time-Series Convergence"""
    rounds = range(len(history))
    high_pcts = [h['high_work_pct'] for h in history]
    low_pcts = [h['low_work_pct'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, high_pcts, label=r'High-Skill ($\theta_H$) Working', color='blue', linewidth=2)
    plt.plot(rounds, low_pcts, label=r'Low-Skill ($\theta_L$) Working', color='red', linewidth=2)
    
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Agent Learning: Convergence to Elite Rat Race Equilibrium', fontsize=14)
    plt.xlabel('Simulation Round', fontsize=12)
    plt.ylabel('Proportion of Group Exerting Effort (e=1)', fontsize=12)
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visual_1_convergence.png')
    plt.show()

def plot_professor_heatmap():
    """Visual 2: The Professor's Heatmap (Optimal Policy Space)"""
    T_A_range = np.linspace(3.0, 7.0, 40)
    T_B_range = np.linspace(2.0, 6.0, 40)
    
    W_matrix = np.zeros((len(T_A_range), len(T_B_range)))
    
    for i, t_a in enumerate(T_A_range):
        for j, t_b in enumerate(T_B_range):
            if t_b >= t_a:
                W_matrix[i, j] = np.nan # T_B cannot be higher than T_A
                continue
                
            e_H_star = theoretical_best_response(PARAMS['theta_H'], PARAMS['C_H'], t_a, t_b)
            e_L_star = theoretical_best_response(PARAMS['theta_L'], PARAMS['C_L'], t_a, t_b)
            
            # W = w_H * E[e_H] + w_L * E[e_L]
            W = PARAMS['omega_H'] * e_H_star + PARAMS['omega_L'] * e_L_star
            W_matrix[i, j] = W

    plt.figure(figsize=(8, 7))
    cmap = plt.cm.viridis
    cmap.set_bad(color='lightgray') # Color for invalid parameter combinations
    
    im = plt.imshow(W_matrix, extent=[T_B_range.min(), T_B_range.max(), T_A_range.min(), T_A_range.max()],
                    origin='lower', aspect='auto', cmap=cmap)
    
    plt.colorbar(im, label=r"Professor's Objective Utility ($W$)")
    plt.title("Professor's Policy Space: Mapping Equilibrium Utility", fontsize=14)
    plt.xlabel(r"B-Grade Threshold ($T_B$)", fontsize=12)
    plt.ylabel(r"A-Grade Threshold ($T_A$)", fontsize=12)
    
    # Mark the placeholder thresholds used in the ABS
    plt.scatter(PARAMS['T_B'], PARAMS['T_A'], color='red', marker='*', s=150, 
                label='Current Simulation Parameters')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visual_2_heatmap.png')
    plt.show()

def create_agent_grid_gif(grid_history):
    """Visual 3: Dynamic Agent Grid GIF"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Custom colormap: 0 = Red (Shirk), 1 = Green (Work)
    cmap = ListedColormap(['#ff4c4c', '#4cff4c'])
    
    im = ax.imshow(grid_history[0], cmap=cmap, vmin=0, vmax=1)
    
    ax.set_title("Agent Decisions Over Time\nTop: High-Skill | Bottom: Low-Skill", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a custom legend
    import matplotlib.patches as mpatches
    work_patch = mpatches.Patch(color='#4cff4c', label='Work (e=1)')
    shirk_patch = mpatches.Patch(color='#ff4c4c', label='Shirk (e=0)')
    ax.legend(handles=[work_patch, shirk_patch], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    def update(frame):
        im.set_array(grid_history[frame])
        ax.set_title(f"Agent Decisions - Round {frame}\nTop: High-Skill | Bottom: Low-Skill", fontsize=12)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=100, blit=True)
    
    # Save as GIF
    ani.save('visual_3_agent_grid.gif', writer='pillow', fps=10)
    print("GIF successfully saved as 'visual_3_agent_grid.gif'")
    plt.close()

# ==========================================
# 5. EXECUTE SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Plot the analytical heatmap to justify threshold choices
    plot_professor_heatmap()
    
    # 2. Run the Simulation
    sim = ClassroomABS()
    sim.run()
    
    # 3. Plot convergence metrics
    plot_convergence(sim.effort_history)
    
    # 4. Generate the GIF animation
    create_agent_grid_gif(sim.grid_history)