import torch
import random
import numpy as np

# -------------------------------------------------
# 🖥️ DEVICE SETUP
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Optional speed boost for NVIDIA GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# -------------------------------------------------
# 🎲 REPRODUCIBILITY
# -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------
# 🤖 ENVIRONMENT SETTINGS
# -------------------------------------------------
NUM_ROBOTS = 5
NUM_PATIENTS = 1000
TIMESTEPS_PER_SESSION = 50
STATE_DIM = 6  # [skill, avg_effort, fatigue, motivation, t_norm, remaining_norm]
ACTION_DIM = NUM_ROBOTS


# -------------------------------------------------
# 🧠 MODEL ARCHITECTURE
# -------------------------------------------------
HIDDEN_1 = 128
HIDDEN_2 = 256
DROPOUT = 0.27


# -------------------------------------------------
# 🎯 RL TRAINING
# -------------------------------------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
TARGET_UPDATE_FREQ = 200  # steps


STATE_DIM = 6
ACTION_DIM = NUM_ROBOTS


# Add experiment parameters
PATIENT_COUNTS = [1000, 2000, 5000, 10000]  # total simulated patients
ARCHITECTURES = [(64, 128), (128, 256), (256, 512)]  # hidden1, hidden2
HYPERPARAMS = [
    {"LR": 1e-3, "BATCH_SIZE": 64, "LOCAL_EPOCHS": 50},
    {"LR": 5e-4, "BATCH_SIZE": 128, "LOCAL_EPOCHS": 100},
    {"LR": 1e-4, "BATCH_SIZE": 32, "LOCAL_EPOCHS": 150},
]

REWARD_SCENARIOS = [
    {"w1": 2.0, "w2": -0.3, "w3": 0.3},  # Skill focus
    {"w1": 1.0, "w2": -1.0, "w3": 0.2},  # Fatigue focus
    {"w1": 1.0, "w2": -0.2, "w3": 1.0},  # Motivation focus
]


# -------------------------------------------------
# 🌍 FEDERATED LEARNING
# -------------------------------------------------
LOCAL_EPOCHS = 3
FED_ROUNDS = 5
CLIENT_VARIANTS = ["A", "B", "C", "D"]

AGGREGATION = "weighted"  # "mean" or "weighted"
CLIENT_FRACTION = 1.0  # % of clients sampled each round

MAX_LOCAL_STEPS = 5
EVAL_EPISODES = 5


# -------------------------------------------------
# 🔒 PRIVACY (for tradeoff experiment)
# -------------------------------------------------
DP_NOISE_STD = 0.0  # changed dynamically during experiments


# -------------------------------------------------
# 🎁 REWARD WEIGHTS (for sensitivity experiments)
# reward = w1*success + w2*fatigue + w3*motivation
# -------------------------------------------------
REWARD_WEIGHTS = (1.5, -0.5, 0.5)


# -------------------------------------------------
# 📊 EVALUATION
# -------------------------------------------------
EVAL_EPISODES = 200
EVAL_RENDER = False


# -------------------------------------------------
# 📈 LOGGING / PLOTS
# -------------------------------------------------
VERBOSE_TRAINING = True
SAVE_MODELS = True
MODEL_SAVE_PATH = "saved_models/"
RESULTS_PATH = "results/"


# -------------------------------------------------
# ⚙️ SCALABILITY EXPERIMENT DEFAULTS
# -------------------------------------------------
MIN_CLIENTS_FOR_CONVERGENCE = 2
MAX_CLIENTS_FOR_TEST = 8
