import numpy as np
import config


class Patient:
    def __init__(
        self, init_skill=None, fatigue_rate=1.0, motivation_rec=1.0, noise_range=2
    ):
        # 1. Use specific skill if provided (from the fixed pool), otherwise random
        if init_skill is not None:
            self.skill = init_skill
        else:
            self.skill = np.random.uniform(0.2, 0.5)

        self.fatigue = 0.2
        self.motivation = 0.7
        self.effort_hist = []

        self.fatigue_rate = fatigue_rate
        self.motivation_rec = motivation_rec
        self.noise_range = noise_range

    def step(self, difficulty):
        # 2. Add noise to simulation
        noise = np.random.uniform(-self.noise_range, self.noise_range) * 0.01

        # 3. Calculate Success (clipped between 0 and 1)
        success = np.clip(self.skill - difficulty + noise, 0, 1)

        # 4. Calculate Effort based on difficulty and fatigue
        effort = difficulty * (1 + self.fatigue)
        self.effort_hist.append(effort)

        # 5. Update Internal State
        self.fatigue += 0.05 * effort * self.fatigue_rate
        self.motivation += 0.03 * (success - 0.5) * self.motivation_rec
        self.motivation = np.clip(self.motivation, 0, 1)

        # Skill improves if successful and motivated
        self.skill += 0.02 * success * self.motivation
        self.skill = np.clip(self.skill, 0, 1)

        return success, effort


class RehabEnv:
    def __init__(self, variant="A"):
        self.variant = variant

        # --- DYNAMIC CONFIGURATION ---
        # Pulls directly from config.py to ensure the sweep works
        self.total_patients = getattr(config, "NUM_PATIENTS", 1000)
        self.max_timesteps = getattr(config, "TIMESTEPS_PER_SESSION", 12)
        self.num_robots = getattr(config, "NUM_ROBOTS", 5)

        # Generate the fixed pool of patients for this experiment
        self.patient_pool = self._generate_patient_pool()

        self.reset()

    def _generate_patient_pool(self):
        """Pre-generates a fixed list of patient skills to sample from."""
        pool = []
        for _ in range(self.total_patients):
            # We store the initial skill so this "person" is consistent across episodes
            profile = {"init_skill": np.random.uniform(0.2, 0.5)}
            pool.append(profile)
        return pool

    def _make_patient(self, profile):
        """Creates a patient using a pre-defined profile."""
        init_skill = profile["init_skill"]

        # Apply client variants (A, B, C, D) dynamically
        if self.variant == "A":
            return Patient(init_skill=init_skill)
        if self.variant == "B":
            return Patient(init_skill=init_skill, noise_range=5)
        if self.variant == "C":
            return Patient(init_skill=init_skill, fatigue_rate=1.5)
        if self.variant == "D":
            return Patient(init_skill=init_skill, motivation_rec=0.5)

        return Patient(init_skill=init_skill)

    def reset(self):
        # Sample from the fixed pool instead of generating new random
        patient_idx = np.random.randint(self.total_patients)
        profile = self.patient_pool[patient_idx]

        self.patient = self._make_patient(profile)
        self.timestep = 0
        return self._state()

    def _state(self):
        # Returns [skill, avg_effort, fatigue, motivation, t_norm, remaining_norm]
        return np.array(
            [
                self.patient.skill,
                np.mean(self.patient.effort_hist) if self.patient.effort_hist else 0.0,
                self.patient.fatigue,
                self.patient.motivation,
                self.timestep / self.max_timesteps,
                (self.max_timesteps - self.timestep) / self.max_timesteps,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        # Calculate difficulty based on Action / Num_Robots
        difficulty = action / self.num_robots
        success, effort = self.patient.step(difficulty)

        # --- DYNAMIC REWARD WEIGHTS ---
        # Uses config.REWARD_WEIGHTS. If config is updated during a sweep, this updates instantly.
        w1, w2, w3 = getattr(config, "REWARD_WEIGHTS", (1.5, -0.5, 0.5))

        reward = (
            w1 * (success - 0.5)
            + w2 * self.patient.fatigue
            + w3 * self.patient.motivation
        )

        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        return self._state(), reward, done
