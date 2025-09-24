import os
import sys
import random
import math
import subprocess
import numpy as np
import traci
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from collections import deque
from traci import trafficlight
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

import random

GLOBAL_SEED = 41  # Change this to 40,41,42 for a different run

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

class QMixer:
    def __init__(self, num_agents, state_dim, embed_dim=32):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        state_input = Input(shape=(state_dim,))
        q_values_input = Input(shape=(num_agents,))

        x = Concatenate()([state_input, q_values_input])
        x = Dense(embed_dim, activation='relu')(x)
        x = Dense(embed_dim, activation='relu')(x)
        q_tot_output = Dense(1, activation='linear')(x)

        self.model = Model(inputs=[state_input, q_values_input], outputs=q_tot_output)
        self.model.compile(optimizer=Adam(0.001), loss='mse')


# === QMIX-compatible replay ===
def qmix_replay(agents, memory, mixer, batch_size=32, gamma=0.99):
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    num_agents = len(agents)
    state_size = agents[0].state_size  

    joint_states = []
    joint_actions = []
    joint_rewards = []
    joint_next_states = []
    dones = []

    for sample in minibatch:
        states = []
        next_states = []

        for s in sample[0]:
            padded = np.full(state_size, 0.0, dtype=np.float32)
            padded[:len(s)] = s[:state_size]
            states.append(padded)

        for ns in sample[3]:
            padded = np.full(state_size, 0.0, dtype=np.float32)
            padded[:len(ns)] = ns[:state_size]
            next_states.append(padded)

        joint_states.append(states)
        joint_next_states.append(next_states)
        joint_actions.append(sample[1])
        joint_rewards.append(sample[2])
        dones.append(sample[4])

    joint_states = np.array(joint_states)
    joint_next_states = np.array(joint_next_states)
    joint_actions = np.array(joint_actions)
    joint_rewards = np.array(joint_rewards)
    dones = np.array(dones).astype(np.float32)

    B, N, S = joint_states.shape
    q_values = np.zeros((B, N))
    next_q_values = np.zeros((B, N))

    for i, agent in enumerate(agents):
        agent_q = agent.model.predict(joint_states[:, i], verbose=0)
        agent_next_q = agent.target_model.predict(joint_next_states[:, i], verbose=0)
        actions = joint_actions[:, i]

        q_values[:, i] = agent_q[np.arange(B), actions]
        next_q_values[:, i] = np.max(agent_next_q, axis=1)

    joint_states_flat = joint_states.reshape(B, -1)
    joint_next_states_flat = joint_next_states.reshape(B, -1)

    q_total = mixer.model.predict([joint_states_flat, q_values], verbose=0)
    next_q_total = mixer.model.predict([joint_next_states_flat, next_q_values], verbose=0)

    target_q_total = joint_rewards.sum(axis=1, keepdims=True) + gamma * next_q_total * (1 - dones[:, None])

    mixer.model.fit([joint_states_flat, q_values], target_q_total, verbose=0)



class CentralTrainer:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.gamma = 0.95
        self.tau = 0.05
        self.update_target()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        self.target_model.set_weights([
            self.tau * w + (1 - self.tau) * tw for w, tw in zip(weights, target_weights)
        ])

    def train(self, memory, batch_size=32):
        minibatch = random.sample(memory, batch_size)

        print("🔍 Verifying joint state shapes in minibatch...")
        for i, sample in enumerate(minibatch):
            try:
                state = sample[0]
                print(f"Sample {i}: type={type(state)}, shape={np.array(state).shape}, len={len(state)}")
            except Exception as e:
                print(f"Sample {i}: ERROR reading state → {e}")

        states = np.array([x[0] for x in minibatch])
        rewards = np.array([x[1] for x in minibatch])
        next_states = np.array([x[2] for x in minibatch])
        dones = np.array([x[3] for x in minibatch])
        actions = np.array([x[4] for x in minibatch])

        q_values = self.model.predict(states, verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(q_next_online[i])
                q_values[i][actions[i]] = rewards[i] + self.gamma * q_next_target[i][best_action]

        self.model.fit(states, q_values, verbose=0)
        self.update_target()

class QMIXTrafficAgent:
    def __init__(self, tls_id, green_phases, yellow_phases, detectors,
                 min_green, max_green, yellow_duration, phase_detector_map=None,
                 exit_detectors=None, exit_detector_map=None):
        self.tls_id = tls_id
        self.green_phases = green_phases
        self.yellow_phases = yellow_phases
        self.yellow_duration = yellow_duration
        self.all_red_phase = 4
        self.step_count = 0
        self.detectors = detectors
        self.exit_detectors = exit_detectors or []
        self.min_green = min_green
        self.max_green = max_green

        self.base_min_green = min_green
        self.base_max_green = max_green

        # --- Per-phase adaptive state ---
        self.phase_min = {i: self.base_min_green for i in range(len(self.green_phases))}
        self.phase_max = {i: self.base_max_green for i in range(len(self.green_phases))}
        self.queue_ema = {i: 0.0 for i in range(len(self.green_phases))}
        self.prev_queue_raw = {i: 0.0 for i in range(len(self.green_phases))}
        self.occ_ema = {i: 0.0 for i in range(len(self.green_phases))}

        #Initialize Hyperparams for responsiveness
        self.ema_alpha = 0.35        # higher = faster reaction
        self.occ_alpha = 0.5
        self.max_step_up = 6         # cap bound increase per update (seconds)
        self.max_step_down = 8       # cap bound decrease per update (seconds)
        self.burst_delta = 4         # vehicles; treat as sudden spike if Δqueue >= this
        self.burst_boost = 8         # temporary extra seconds on max when burst detected
        self.empty_decay = 0.7       # shrink factor when approach is empty
        self.hard_min = 5
        self.hard_max = 90

        self.phase_detector_map = phase_detector_map or {}
        self.all_detectors = sum(self.phase_detector_map.values(), [])

        self.action_size = len(green_phases)
        self.predicted_durations = {i: [] for i in range(self.action_size)}
        self.exit_detector_map = exit_detector_map or {}

        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.batch_size = 16
        self.tau = 0.05

        self.phase_index = 0
        self.phase_state = "green"
        self.phase_timer = 0
        self.cycle_phase_counter = 0
        self.ready_for_replay = False

        self.vehicle_wait_times = {}
        self.phase_exit_vehicles = set()
        self.exit_vehicle_seen = set()

        self.total_queue_length_sum = 0.0
        self.queue_measurements = 0
        self.total_vehicle_wait_time = 0.0
        self.vehicle_count = 0
        self.step_throughput_100 = 0
        self.cumulative_avg_queue = 0
        self.cumulative_avg_wait = 0
        self.cumulative_throughput = 0
        self.cumulative_co2 = 0.0

        self.queue_sum_100 = 0.0
        self.queue_count_100 = 0
        self.wait_sum_100 = 0.0
        self.vehicle_count_100 = 0
        self.reward_sum_100 = 0.0

        self.avg_queue_history = []
        self.avg_wait_history = []
        self.throughput_100_history = []
        self.cumulative_avg_queue_history = []
        self.cumulative_avg_wait_history = []
        self.cumulative_throughput_history = []
        self.cumulative_co2_history = []

    def _build_model(self):
        if not hasattr(self, "state_size"):
            raise ValueError(f"state_size not set for {self.tls_id}. You must call initialize_after_traci() first.")

        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def compute_state_size(self):
        local_state_size = len(self.green_phases) * 2  # queue + wait per phase
        one_hot_size = len(self.green_phases)
        neighbor_state_size = 0
        if hasattr(self, "neighbor_agents"):
            neighbor_state_size = 2 * len(self.neighbor_agents)  # queue + wait per neighbor
        return local_state_size + one_hot_size + neighbor_state_size


    def initialize_after_traci(self):
        base_state_size = 2 * len(self.green_phases) + len(self.green_phases)
        num_neighbors = len(NEIGHBOR_MAP[self.tls_id])
        self.state_size = base_state_size + 2 * num_neighbors  # 2 values per neighbor (queue + wait)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def get_state(self):
        state = []
        max_queue = 30.0
        max_wait = 60.0

        # === Local phase-related state ===
        for phase in self.green_phases:
            detectors = self.phase_detector_map.get(phase, [])
            if not detectors:
                print(f"[WARNING] No detectors mapped for phase {phase} on {self.tls_id}")

            try:
                q = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)
            except traci.TraCIException:
                q = 0.0

            w = 0.0
            try:
                lanes = [traci.lanearea.getLaneID(d) for d in detectors if traci.lanearea.getLaneID(d)]
                wait_times = []
                for l in lanes:
                    try:
                        if traci.lane.getLastStepVehicleNumber(l) > 0:
                            wait_times.append(traci.lane.getWaitingTime(l))
                    except:
                        continue
                w = np.mean(wait_times) if wait_times else 0.0
            except traci.TraCIException:
                w = 0.0

            state.append(min(q / max_queue, 1.0))
            state.append(min(w / max_wait, 1.0))

        # === Phase one-hot ===
        phase_one_hot = [0] * len(self.green_phases)
        phase_one_hot[self.phase_index] = 1
        state.extend(phase_one_hot)

        # === Neighbor state (if agent.neighbor_agents is set externally) ===
        if hasattr(self, "neighbor_agents"):
            for neighbor in self.neighbor_agents:
                try:
                    q = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in neighbor.all_detectors)
                except:
                    q = 0.0
                try:
                    lanes = [traci.lanearea.getLaneID(d) for d in neighbor.all_detectors if traci.lanearea.getLaneID(d)]
                    wait_times = []
                    for l in lanes:
                        try:
                            if traci.lane.getLastStepVehicleNumber(l) > 0:
                                wait_times.append(traci.lane.getWaitingTime(l))
                        except:
                            continue
                    w = np.mean(wait_times) if wait_times else 0.0
                except:
                    w = 0.0
                state.extend([min(q / max_queue, 1.0), min(w / max_wait, 1.0)])

        # === Pad or clip to match state_size ===
        if len(state) < self.state_size:
            state.extend([0.0] * (self.state_size - len(state)))
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        return np.array(state, dtype=np.float32)


    def get_neighbor_state(self, neighbor_agents):
        neighbor_state = []
        for neighbor in neighbor_agents:
            if neighbor is self:
                continue
            # Example: include current queue of neighbor
            q = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in neighbor.all_detectors)
            w = np.mean([
                traci.lane.getWaitingTime(traci.lanearea.getLaneID(d))
                for d in neighbor.all_detectors
                if traci.lanearea.getLaneID(d) and traci.lane.getLastStepVehicleNumber(traci.lanearea.getLaneID(d)) > 0
            ]) if neighbor.all_detectors else 0.0
            neighbor_state.extend([q / 30.0, w / 60.0])  # normalized
        return np.array(neighbor_state, dtype=np.float32)

    def remember(self, state, reward, next_state, done, phase_index):
        if np.count_nonzero(state) == 0:
            print(f"[SKIP][{self.tls_id}] Ignoring empty state with reward={reward}")
            return
        if reward <= -100:
            print(f"[SKIP][{self.tls_id}] Ignoring severely negative reward: {reward}")
        if state.shape[0] != self.state_size or next_state.shape[0] != self.state_size:
            print(f"[{self.tls_id}] Skipping inconsistent memory entry: got {state.shape[0]}, expected {self.state_size}")
            return
        self.memory.append((state, reward, next_state, done, phase_index))

    def _update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        self.target_model.set_weights([
            self.tau * w + (1 - self.tau) * tw
            for w, tw in zip(weights, target_weights)
        ])

    def adjust_phase_bounds(self, phase_index):
        """
        Per-phase adaptive bounds using EMA of queue, Δqueue burst, and occupancy.
        Called right before that phase turns green.
        """
        dets = self.phase_detector_map.get(phase_index, [])
        if not dets:
            # keep existing bounds if no detectors
            return

        # Raw measurements
        queue_raw = 0.0
        occ_raw = 0.0
        for d in dets:
            try:
                queue_raw += traci.lanearea.getLastStepVehicleNumber(d)
                occ_raw += traci.lanearea.getLastStepOccupancy(d)  # 0..100
            except traci.TraCIException:
                continue

        lanes = max(1, len(dets))
        # Normalize crude "capacity" to lanes; treat > ~4 veh/lane as heavy
        max_possible_queue = 4.0 * lanes
        congestion = 0.0 if max_possible_queue == 0 else min(queue_raw / max_possible_queue, 1.0)

        # EMA updates
        q_old = self.queue_ema[phase_index]
        self.queue_ema[phase_index] = (1 - self.ema_alpha) * q_old + self.ema_alpha * queue_raw

        o_old = self.occ_ema[phase_index]
        self.occ_ema[phase_index] = (1 - self.occ_alpha) * o_old + self.occ_alpha * (occ_raw / (100.0 * lanes))

        # Burst detector (fast rise)
        dq = queue_raw - self.prev_queue_raw[phase_index]
        self.prev_queue_raw[phase_index] = queue_raw
        burst = dq >= self.burst_delta

        # Target bounds (start from base)
        t_min = self.base_min_green
        t_max = self.base_max_green

        # Scale with congestion & occupancy EMA
        # Aggressive upscaling when busy; gentle when light
        up_scale = 1.0 + 0.6 * congestion + 0.5 * self.occ_ema[phase_index]
        down_scale = 1.0 - 0.5 * (1.0 - congestion)  # faster shrink when light

        # Empty approach? shrink quickly (with a floor)
        if queue_raw <= 0.5:
            t_min = max(self.hard_min, int(t_min * self.empty_decay))
            t_max = max(t_min + 1, int(t_max * self.empty_decay))
        else:
            t_min = int(t_min * down_scale)
            t_max = int(t_max * up_scale)

        # Burst boost: temporarily allow a bigger max to flush sudden jams
        if burst:
            t_max = min(self.hard_max, t_max + self.burst_boost)

        # Cap absolute bounds
        t_min = max(self.hard_min, min(t_min, self.hard_max - 1))
        t_max = max(t_min + 1, min(t_max, self.hard_max))

        # Smooth per-update change to avoid oscillation
        cur_min = self.phase_min[phase_index]
        cur_max = self.phase_max[phase_index]

        if t_min > cur_min:
            cur_min = min(t_min, cur_min + self.max_step_up)
        else:
            cur_min = max(t_min, cur_min - self.max_step_down)

        if t_max > cur_max:
            cur_max = min(t_max, cur_max + self.max_step_up)
        else:
            cur_max = max(t_max, cur_max - self.max_step_down)

        # Commit per-phase bounds
        self.phase_min[phase_index] = cur_min
        self.phase_max[phase_index] = max(cur_min + 1, cur_max)

        # Also update the agent-level values so downstream code keeps working
        self.min_green = self.phase_min[phase_index]
        self.max_green = self.phase_max[phase_index]

        print(f"[{self.tls_id}] Phase {phase_index}: q={queue_raw:.1f} dQ={dq:.1f} occ={occ_raw:.1f}% "
            f"→ min={self.min_green} max={self.max_green} (EMAq={self.queue_ema[phase_index]:.2f})")
        
    def log_metrics(self):
        avg_queue = self.queue_sum_100 / self.queue_count_100 if self.queue_count_100 else 0
        avg_wait = self.wait_sum_100 / self.vehicle_count_100 if self.vehicle_count_100 else 0

        self.cumulative_avg_queue = self.total_queue_length_sum / self.queue_measurements if self.queue_measurements else 0
        self.cumulative_avg_wait = self.total_vehicle_wait_time / self.vehicle_count if self.vehicle_count else 0

        # Save to history
        self.avg_queue_history.append(avg_queue)
        self.avg_wait_history.append(avg_wait)
        self.throughput_100_history.append(self.step_throughput_100)

        self.cumulative_avg_queue_history.append(self.cumulative_avg_queue)
        self.cumulative_avg_wait_history.append(self.cumulative_avg_wait)
        self.cumulative_throughput_history.append(self.cumulative_throughput)
        self.cumulative_co2_history.append(self.cumulative_co2)

    def print_metrics(self, step, epsilon, q_values):
        avg_queue = self.queue_sum_100 / self.queue_count_100 if self.queue_count_100 else 0
        avg_wait = self.wait_sum_100 / self.vehicle_count_100 if self.vehicle_count_100 else 0

        print(f"[DQN][{self.tls_id}][Step {step}] Avg Queue: {avg_queue:.2f} | Avg Wait: {avg_wait:.2f}s | "
              f"Throughput(100): {self.step_throughput_100} | Total: {self.cumulative_throughput} | "
              f"Cum Avg Queue: {self.cumulative_avg_queue:.2f} | Cum Avg Wait: {self.cumulative_avg_wait:.2f}s | ε: {epsilon:.4f} | ")
        
        #print(f"Avg Reward(100): {self.reward_sum_100:.4f} | Avg Durations: {avg_durations}")

    def reset_step_metrics(self):
        self.queue_sum_100 = 0.0
        self.queue_count_100 = 0
        self.wait_sum_100 = 0.0
        self.vehicle_count_100 = 0
        self.step_throughput_100 = 0
        self.reward_sum_100 = 0.0
        for a in self.predicted_durations:
            self.predicted_durations[a] = []

def centralized_replay(agents, central_memory, batch_size=32, gamma=0.95):
        if len(central_memory) < batch_size:
            return
        minibatch = random.sample(central_memory, batch_size)

        for i, agent in enumerate(agents):
            states = np.array([transition[0][i] for transition in minibatch])
            actions = [transition[1][i] for transition in minibatch]
            rewards = [transition[2][i] for transition in minibatch]
            next_states = np.array([transition[3][i] for transition in minibatch])

            q_values = agent.model.predict(states, verbose=0)
            q_next_online = agent.model.predict(next_states, verbose=0)
            q_next_target = agent.target_model.predict(next_states, verbose=0)

            for j in range(batch_size):
                a = actions[j]
                if rewards[j] <= -100:
                    q_values[j][a] = rewards[j]
                else:
                    next_a = np.argmax(q_next_online[j])
                    q_values[j][a] = rewards[j] + gamma * q_next_target[j][next_a]

            agent.model.fit(states, q_values, verbose=0)
            agent._update_target_model()


def epsilon_by_step(step, total_steps, warmup_steps=10000, epsilon_min=0.01, epsilon_start=1.0, epsilon_target=0.1):
    if step < warmup_steps:
        return 1.0  # full exploration
    decay_total = total_steps - warmup_steps
    k = math.log(epsilon_start / epsilon_target) / decay_total
    epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-k * (step - warmup_steps))
    return max(epsilon, epsilon_min)

def compute_phase_reward(agent, idx, step, duration, base_min, base_max, hard_max=90, clearance_weight=1.8):

        # --- Measure current queue for this phase ---
        phase_detectors = agent.phase_detector_map.get(idx, [])
        queue = 0
        throughput = 0
        emission_penalty = 0.0
        vehicle_count = 0

        try:
            for det in phase_detectors:
                queue += traci.lanearea.getLastStepHaltingNumber(det)
            # Throughput from exit detectors
            exit_dets = agent.exit_detector_map.get(idx, [])
            for det in exit_dets:
                throughput += traci.lanearea.getLastStepVehicleNumber(det)
            # Emissions
            for det in phase_detectors:
                lane_id = traci.lanearea.getLaneID(det)
                if lane_id:
                    for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                        co2_mg_s = traci.vehicle.getCO2Emission(vid)  # mg/s
                        emission_penalty += co2_mg_s / 1000.0  # → grams/s
                        vehicle_count += 1
        except traci.TraCIException:
            pass

        # --- Normalize emissions ---
        emission_penalty *= duration  # total grams over phase
        if vehicle_count > 0:
            emission_penalty /= vehicle_count  # per-vehicle average
        norm_emission_penalty = 0.002 * emission_penalty  # small weight

        # --- Queue clearance benefit (real clearance) ---
        if not hasattr(agent, "prev_phase_queue"):
            agent.prev_phase_queue = {p: 0 for p in agent.green_phases}
        #cleared = max(prev_queue - queue, 0)
        queue_clearance_benefit = queue * clearance_weight
        #agent.prev_phase_queue[idx] = queue

        # --- Fairness bonus ---
        if not hasattr(agent, "last_served_step"):
            agent.last_served_step = {p: 0 for p in agent.green_phases}
        phase_key = agent.green_phases[idx]
        time_since_served = step - agent.last_served_step[phase_key]
        fairness_bonus = min(time_since_served / 100.0, 2.0)
        agent.last_served_step[phase_key] = step

        # --- Congestion-aware dynamic bounds ---
        overload_threshold = 2 * len(agent.green_phases)
        congestion_level = min(queue / overload_threshold, 1.0)
        dyn_max = min(base_max + int((base_max - base_min) * congestion_level), hard_max)
        dyn_min = base_min

        # Duration alignment reward (if action matches congestion-aware bound)
        if congestion_level > 0.7 and duration < dyn_max:
            duration_alignment_bonus = - (dyn_max - duration) * 0.1
        elif congestion_level < 0.3 and duration > dyn_min:
            duration_alignment_bonus = - (duration - dyn_min) * 0.1
        else:
            duration_alignment_bonus = 0

        # --- Base reward ---
        throughput_reward = 3.5 * throughput
        queue_penalty = queue * (0.05 if queue > overload_threshold else 0)

        if queue == 0 and throughput == 0:
            reward = -100  # light penalty for idle phase
        else:
            reward = (
                throughput_reward
                + queue_clearance_benefit
                + fairness_bonus
                + duration_alignment_bonus
                - queue_penalty
                - norm_emission_penalty
            )

        # --- Debug logging ---
        print(
            f"[Step {step}] [{agent.tls_id}] Phase {idx} |"
            f"Reward {reward:.3f} | Throughput {throughput} | Queue {queue} |"
            f"Clearance  {queue_clearance_benefit} | Fair {fairness_bonus:.2f} |"
            f"Duration Bonus {duration_alignment_bonus:.2f} |"
            f"CO2pen {norm_emission_penalty:.3f} |"
            f"Bounds [{dyn_min}, {dyn_max}] |  Duration {duration}"
        )

        return reward


def scale_q_values_to_durations(q_values, min_green, max_green):
    q_values = np.array(q_values, dtype=np.float64)

    q_min = np.min(q_values)
    q_max = np.max(q_values)

    # Avoid division by zero
    if q_max - q_min < 1e-6:
        durations = [min_green] * len(q_values)
        durations[np.argmax(q_values)] = max_green
        return durations

    # Normalize to [0, 1] where min Q = 0, max Q = 1
    q_normalized = (q_values - q_min) / (q_max - q_min)

    durations = [
        int(min_green + qn * (max_green - min_green))
        for qn in q_normalized
    ]

    return durations


global_metrics = {
    "cum_avg_queue": [],
    "cum_avg_wait": [],
    "cum_node_throughput": [],
    "cum_node_co2": [],
    "cum_network_throughput": []  
}

def plot_agent_metrics(agents):
    step_interval = 100

    # 1. Avg Queue Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.avg_queue_history))]
        plt.plot(steps, agent.avg_queue_history, marker='o', label=agent.tls_id)
    plt.title("CTDE-QMIX Average Queue per 100 Steps")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Queue Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Avg Wait Time Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.avg_wait_history))]
        plt.plot(steps, agent.avg_wait_history, marker='o', label=agent.tls_id)
    plt.title("CTDE-QMIX Average Wait Time per 100 Steps")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Wait (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Throughput per 100 Steps Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.throughput_100_history))]
        plt.plot(steps, agent.throughput_100_history, marker='o', label=agent.tls_id)
    plt.title("CTDE-QMIX Throughput per 100 Steps")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Vehicles Exited (100-step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

   # 4. Cumulative queue
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.cumulative_avg_queue_history))]
        plt.plot(steps, agent.cumulative_avg_queue_history, marker='o', label=agent.tls_id)
    plt.title("CTDE-QMIX Cumulative Average Queue Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Average Queue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Cumulative wait
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.cumulative_avg_wait_history))]
        plt.plot(steps, agent.cumulative_avg_wait_history, marker='o', label=agent.tls_id)
    plt.title("CTDE-QMIX Cumulative Average Wait Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Average Wait")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Cumulative Throughput Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.cumulative_throughput_history))]
        plt.plot(steps, agent.cumulative_throughput_history, label=agent.tls_id)
    plt.title("CTDE-QMIX Cumulative Throughput Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Vehicles Exited")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

            # 4. Cumulative Throughput Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.cumulative_co2_history))]
        plt.plot(steps, agent.cumulative_co2_history, label=agent.tls_id)
    plt.title("CTDE-QMIX Cumulative Co2 Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Co2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def plot_global_metrics(cum_avg_queue, cum_avg_wait, cum_node_throughput,cum_node_co2):
    step_interval = 100  # Logging interval in simulation steps
    steps = [i * step_interval for i in range(len(cum_avg_queue))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_avg_queue, label='CTDE-QMIX Cumulative Avg Queue', color='purple')
    plt.title("CTDE-QMIX Global Cumulative Average Queue Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Queue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len( cum_avg_wait))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_avg_wait, label='CTDE-QMIX Cumulative Avg Wait', color='brown')
    plt.title("CTDE-QMIX Global Cumulative Average Wait Time Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Wait (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len(cum_node_throughput))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_node_throughput, label='CTDE-QMIX Cumulative Node Throughput', color='darkgreen')
    plt.title("CTDE-QMIX Global Cumulative Throughput (Node Sum) Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len(cum_node_co2))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_node_co2, label='CTDE-QMIX Cumulative Node CO2', color='darkgreen')
    plt.title("CTDE-QMIX Global Cumulative Co2 (Node Sum) Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Co2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

NEIGHBOR_MAP = {
    "Node1": ["Node2"],
    "Node2": ["Node1", "Node3"],
    "Node3": ["Node2", "Node4"],
    "Node4": ["Node3", "Node5"],
    "Node5": ["Node4"]
}

from collections import deque
central_memory = deque(maxlen=10000)


def run_simulation(agents, total_steps=500, warmup_steps=1000):
    import numpy as np
    from collections import deque
    step = 0
    epsilon = 1.0
    LOG_INTERVAL = 100
    REPLAY_INTERVAL = 10
    last_step_logged = 0

    global_throughput_100 = 0
    cumulative_global_throughput = 0
    cumulative_global_co2 =0
    queue_window_history, wait_window_history, throughput_window_history = [], [], []
    central_memory = deque(maxlen=10000)

    # === Initialize each agent ===
    for agent in agents:
        old_logic = traci.trafficlight.getAllProgramLogics(agent.tls_id)[0]
        new_phases = []
        for i, old_phase in enumerate(old_logic.phases):
            if i == agent.phase_index:
                new_phase = traci.trafficlight.Phase(duration=150, minDur=agent.min_green, maxDur=agent.max_green, state=old_phase.state)
            else:
                new_phase = traci.trafficlight.Phase(duration=old_phase.duration, minDur=old_phase.minDur, maxDur=old_phase.maxDur, state=old_phase.state)
            new_phases.append(new_phase)

        logic = traci.trafficlight.Logic(programID="manual_control", type=old_logic.type, currentPhaseIndex=old_logic.currentPhaseIndex, phases=new_phases)
        traci.trafficlight.setProgramLogic(agent.tls_id, logic)

        agent.phase_index = 0
        agent.phase_state = "green"
        agent.phase_timer = agent.min_green
        agent.initialize_after_traci()
        agent.neighbor_agents = [a for a in agents if a.tls_id in NEIGHBOR_MAP[agent.tls_id]]
        agent.current_phase_durations = [agent.min_green for _ in agent.green_phases]
        agent.state = agent.get_state()

    max_state_size = max(agent.compute_state_size() for agent in agents)
    for agent in agents:
        agent.state_size = max_state_size
        agent.model = agent._build_model()           # <- Rebuild with updated state size
        agent.target_model = agent._build_model()
        agent.target_model.set_weights(agent.model.get_weights())

    mixer = QMixer(num_agents=len(agents), state_dim=len(agents[0].get_state()) * len(agents))

    while step < total_steps:
        traci.simulationStep()
        step += 1
        epsilon = epsilon_by_step(step, total_steps)
        arrived = traci.simulation.getArrivedNumber()
        global_throughput_100 += arrived
        cumulative_global_throughput += arrived

        for agent in agents:
            idx = agent.phase_index
            phase = agent.green_phases[idx]

            if agent.phase_timer <= 0:
                phase_detectors = agent.phase_detector_map.get(idx, [])
                if agent.phase_state == "green":
                    throughput, wait_time = 0, 0
                    for vid in agent.phase_exit_vehicles:
                        if vid not in agent.exit_vehicle_seen:
                            agent.exit_vehicle_seen.add(vid)
                            throughput += 1
                            wt = agent.vehicle_wait_times.pop(vid, 0.0)
                            wait_time += wt
                            agent.total_vehicle_wait_time += wt
                            agent.vehicle_count += 1
                            agent.step_throughput_100 += 1
                            agent.wait_sum_100 += wt
                            agent.vehicle_count_100 += 1
                    agent.cumulative_throughput += throughput
                    agent.phase_exit_vehicles.clear()

                    detectors = agent.phase_detector_map[phase]
                    queue = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)
                    avg_wait = wait_time / throughput if throughput > 0 else 0.0

                    duration = agent.current_phase_durations[idx]

                    reward = compute_phase_reward(agent,idx, step, duration, agent.base_min_green, agent.base_max_green) 

                    print(f"[Step {step}] [{agent.tls_id}] Phase {idx} Reward {reward} throughput {throughput} Queue {queue} Durations {agent.current_phase_durations}")

                    try:
                        for det in phase_detectors:
                            lane_id = traci.lanearea.getLaneID(det)
                            if lane_id:
                                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                                    co2_g_s = traci.vehicle.getCO2Emission(vid) / 1000.0  # mg/s → g/s
                                    agent.cumulative_co2 += co2_g_s
                    except traci.TraCIException:
                        pass

                    next_state = agent.get_state()
                    done = step >= total_steps
                    agent.remember(agent.state, reward, next_state, done, idx)
                    agent.state = next_state

                    try:
                        for det in phase_detectors:
                            lane_id = traci.lanearea.getLaneID(det)
                            if lane_id:
                                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                                    co2_g_s = traci.vehicle.getCO2Emission(vid) / 1000.0  # mg/s → g/s
                                    agent.cumulative_co2 += co2_g_s
                    except traci.TraCIException:
                        pass

                    traci.trafficlight.setPhase(agent.tls_id, agent.yellow_phases[idx])
                    agent.phase_state = "yellow"
                    agent.phase_timer = agent.yellow_duration

                elif agent.phase_state == "yellow":
                    try:
                        for det in phase_detectors:
                            lane_id = traci.lanearea.getLaneID(det)
                            if lane_id:
                                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                                    co2_g_s = traci.vehicle.getCO2Emission(vid) / 1000.0  # mg/s → g/s
                                    agent.cumulative_co2 += co2_g_s
                    except traci.TraCIException:
                        pass
                    traci.trafficlight.setPhase(agent.tls_id, agent.all_red_phase)
                    agent.phase_state = "red"
                    agent.phase_timer = agent.all_red_phase

                elif agent.phase_state == "red":
                    try:
                        for det in phase_detectors:
                            lane_id = traci.lanearea.getLaneID(det)
                            if lane_id:
                                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                                    co2_g_s = traci.vehicle.getCO2Emission(vid) / 1000.0  # mg/s → g/s
                                    agent.cumulative_co2 += co2_g_s
                    except traci.TraCIException:
                        pass
                    agent.phase_index = (agent.phase_index + 1) % len(agent.green_phases)
                    agent.state = agent.get_state()

                
                    # Predict Durations
                    agent.adjust_phase_bounds(agent.phase_index)
                    # Predict durations with updated bounds
                    q_values = agent.model.predict(np.array([agent.state]), verbose=0)[0]
                    durations = scale_q_values_to_durations(q_values, agent.min_green, agent.max_green)
                    agent.current_phase_durations = durations
                    agent.q_values = q_values

                    duration = agent.current_phase_durations[agent.phase_index]
                    agent.predicted_durations[agent.phase_index].append(duration)

                    traci.trafficlight.setPhase(agent.tls_id, agent.green_phases[agent.phase_index])
                    agent.phase_state = "green"
                    agent.phase_timer = duration

                    print(f"[{agent.tls_id}][Step {step}]    Green Phase {agent.phase_index} for {duration} steps  Q-values: {getattr(agent, 'q_values', ['?']*agent.action_size)}")

                    agent.cycle_phase_counter += 1
                    if agent.cycle_phase_counter >= len(agent.green_phases):
                        agent.cycle_phase_counter = 0
                        agent.ready_for_replay = True
            else:
                agent.phase_timer -= 1

            if agent.phase_state == "green":
                q = sum(traci.lanearea.getLastStepHaltingNumber(d) for d in agent.all_detectors)
                agent.total_queue_length_sum += q
                agent.queue_measurements += 1
                agent.queue_sum_100 += q
                agent.queue_count_100 += 1
                for d in agent.exit_detector_map.get(phase, []):
                    try:
                        vids = traci.lanearea.getLastStepVehicleIDs(d)
                        for vid in vids:
                            if vid not in agent.exit_vehicle_seen:
                                agent.phase_exit_vehicles.add(vid)
                    except:
                        continue
                for vid in traci.vehicle.getIDList():
                    try:
                        agent.vehicle_wait_times[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    except:
                        continue

        joint_state, joint_next_state, joint_action, joint_reward = [], [], [], []
        for agent in agents:
            s = agent.state
            ns = agent.get_state()
            joint_state.append(s)
            joint_next_state.append(ns)
            joint_action.append(agent.phase_index)
            joint_reward.append(agent.last_reward if hasattr(agent, 'last_reward') else 0.0)
        central_memory.append((joint_state, joint_action, joint_reward, joint_next_state, step >= total_steps))

        if step > warmup_steps and all(agent.ready_for_replay for agent in agents):
            print(f"[REPLAY][Step {step}] All agents completed a cycle. Coordinated Replay now.")
            qmix_replay(agents, central_memory, mixer)
            for agent in agents:
                agent.ready_for_replay = False

        if step - last_step_logged >= LOG_INTERVAL:
            last_step_logged = step
            window_queue_sum, window_wait_sum, window_throughput = 0, 0, 0

            for agent in agents:
                window_queue_sum += agent.queue_sum_100 / agent.queue_count_100 if agent.queue_count_100 else 0
                window_wait_sum += agent.wait_sum_100 / agent.vehicle_count_100 if agent.vehicle_count_100 else 0
                cumulative_global_co2 += agent.cumulative_co2
                window_throughput += agent.step_throughput_100

                agent.log_metrics()
                agent.print_metrics(step, epsilon, agent.q_values if hasattr(agent, "q_values") else [0] * agent.action_size)
                agent.reset_step_metrics()

            avg_queue = window_queue_sum / len(agents)
            avg_wait = window_wait_sum / len(agents)
            cum_avg_queue = (sum(queue_window_history) + avg_queue) / (len(queue_window_history) + 1)
            cum_avg_wait = (sum(wait_window_history) + avg_wait) / (len(wait_window_history) + 1)
            cum_nodes_throughput = sum(throughput_window_history) + window_throughput

            global_metrics["cum_avg_queue"].append(cum_avg_queue)
            global_metrics["cum_avg_wait"].append(cum_avg_wait)
            global_metrics["cum_node_throughput"].append(cum_nodes_throughput)
            global_metrics["cum_node_co2"].append(cumulative_global_co2)
            global_metrics["cum_network_throughput"].append(cumulative_global_throughput)

            print(f" [GLOBAL FIXED][Step {step}] Avg Queue: {avg_queue:.2f} | Avg Wait: {avg_wait:.2f}s | "
                  f"Throughput(100): {window_throughput} | Cum Throughput: {cum_nodes_throughput} |  Cum Co2: {cumulative_global_co2} | "
                  f"Network Throughput(100): {global_throughput_100} | Cum Network Throughput: {cumulative_global_throughput}\n")

            queue_window_history.append(avg_queue)
            wait_window_history.append(avg_wait)
            throughput_window_history.append(window_throughput)
            global_throughput_100 = 0

    print("Simulation completed.")

def main():
    SUMO_CONFIG = [
        'sumo',
        '-c', 'C:/Users/FrancisChiny_mmj9g1h/Downloads/v7 (2)/v7/CenturyCityNode1_V1.sumocfg',
        '--step-length', '1.0',
        '--delay', '1',
        '--lateral-resolution', '0',
        '--ignore-junction-blocker', '2',
        '--no-warnings', 'true',
        '--seed', str(GLOBAL_SEED)
    ]

    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
        
    agents = [
        QMIXTrafficAgent(
            tls_id="Node1",
            green_phases=[0, 1, 2, 3, 4],
            yellow_phases=[0, 1, 2, 3, 4],
            detectors=[
                "Node1_EB_0", "Node1_EB_1", "Node1_EB_2",
                "Node1_SB_0", "Node1_NB_0",
                "Node1_WB_0", "Node1_WB_1", "Node1_WB_2",
                "Node1_EB_L2_0", "Node1_WB_L2_0"
            ],
            exit_detectors=[
                "Node1_NB_Exit_0", "Node1_EB_Exit_1", "Node1_EB_Exit_0",
                "Node1_WB_2_Exit_0", "Node1_EB_L2_0_Exit_0", "Node1_SB_0_Exit_0"
            ],
            min_green=12,
            max_green= 30,
            yellow_duration=4,
            phase_detector_map={
                0: ["Node1_WB_0", "Node1_WB_1", "Node1_WB_2"],
                1: ["Node1_NB_0", "Node1_SB_0"],
                2: ["Node1_WB_L2_0","Node1_NB_0"],             
                3: ["Node1_EB_0", "Node1_EB_1", "Node1_EB_2",],
                4: ["Node1_EB_L2_0"],
                
            },
             exit_detector_map={
                0: ["Node1_SB_0_Exit_0","Node1_NB_Exit_0","Node1_EB_Exit_0","Node1_EB_Exit_1","Node1_EB_L2_0_Exit_0","Node1_WB_2_Exit_0","Node1_WB_2_Exit_1","Node1_WB_L2_Exit_0"],
                1: ["Node1_SB_0_Exit_0","Node1_NB_Exit_0","Node1_EB_Exit_0","Node1_EB_Exit_1","Node1_EB_L2_0_Exit_0","Node1_WB_L2_Exit_0"],
                2: ["Node1_SB_0_Exit_0","Node1_NB_Exit_0","Node1_EB_Exit_0","Node1_EB_Exit_1","Node1_EB_L2_0_Exit_0","Node1_WB_2_Exit_0","Node1_WB_2_Exit_1","Node1_WB_L2_Exit_0"],
                3: ["Node1_SB_0_Exit_0","Node1_NB_Exit_0","Node1_EB_Exit_0","Node1_EB_Exit_1","Node1_EB_L2_0_Exit_0","Node1_WB_2_Exit_0","Node1_WB_2_Exit_1","Node1_WB_L2_Exit_0"],
                4: ["Node1_SB_0_Exit_0","Node1_NB_Exit_0","Node1_EB_L2_0_Exit_0","Node1_WB_2_Exit_0","Node1_WB_2_Exit_1","Node1_WB_L2_Exit_0"],
            }
        ),
        QMIXTrafficAgent(
            tls_id="Node2",
            green_phases=[0, 1, 2, 3],
            yellow_phases=[0, 1, 2, 3],
            detectors=[
                    "Node2_EB_0", "Node2_EB_1", "Node2_EB_2",
                    "Node2_SB_0", "Node2_SB_1",
                    "Node2_NB_0", "Node2_NB_1", "Node2_NB_2",
                    "Node2_WB_0", "Node2_WB_1", "Node2_WB_2",
            ],
            exit_detectors=[
                "Node2_EB_Exit_1", "Node2_EB_Exit_0", "Node2_WB_Exit_0","Node2_WB_Exit_1"
                "Node2_SB_Exit_0", "Node2_NB_Exit_0"
            ],
            min_green=15,
            max_green=35,
            yellow_duration=6,
            phase_detector_map={
                    0: ["Node2_EB_0", "Node2_EB_1", "Node2_EB_2","Node2_WB_0", "Node2_WB_1", "Node2_WB_2"],  # EB lanes for phase 0
                    1: ["Node2_EB_2", "Node2_EB_0","Node2_WB_0","Node2_WB_2"],                 # SB lanes for phase 2
                    2: ["Node2_SB_0", "Node2_NB_0", "Node2_NB_1","Node2_NB_2"],  # NB lanes for phase 4
                    3: ["Node2_SB_1", "Node2_NB_2"],
            },
             exit_detector_map={
                0: ["Node2_WB_Exit_0", "Node2_WB_Exit_1","Node2_EB_Exit_0","Node2_EB_Exit_1","Node2_NB_Exit_0","Node2_SB_Exit_0"],
                1: ["Node2_NB_Exit_0","Node2_SB_Exit_0"],
                2: ["Node2_WB_Exit_0","Node2_WB_Exit_1", "Node2_NB_Exit_0", "Node2_EB_Exit_0","Node2_EB_Exit_1",],
                3: ["Node2_WB_Exit_1","Node2_EB_Exit_1"],  # update if needed
            }
        ),
        QMIXTrafficAgent(
            tls_id="Node3",
            green_phases=[0, 1, 2, 3],
            yellow_phases=[0, 1, 2, 3],
            detectors=[
                       "Node3_EB_0", "Node3_EB_1", "Node3_EB_2", "Node3_EB_3",
                       "Node3_NB_0", "Node3_NB_1", "Node3_NB_2",
                       "Node3_SB_0", "Node3_SB_1", "Node3_SB_2",
                       "Node3_WB_0", "Node3_WB_1", "Node3_WB_2",
            ],
            exit_detectors=[
                        "Node3_EB_0", "Node3_EB_1", "Node3_EB_2", "Node3_EB_3", "Node3_WB_0", "Node3_WB_1", "Node3_WB_2",
                        "Node3_EB_0","Node3_EB_1", "Node3_WB_0", "Node3_WB_2",
                        "Node3_SB_0", "Node3_SB_1", "Node3_SB_2", "Node3_NB_0", "Node3_NB_1", "Node3_NB_2",
                        "Node3_SB_0", "Node3_SB_2", "Node3_NB_0","Node3_NB_1",
            ],
            min_green=15,
            max_green=35,
            yellow_duration=6,
            phase_detector_map={
                        0: ["Node3_EB_0", "Node3_EB_1", "Node3_EB_2", "Node3_EB_3", "Node3_WB_0", "Node3_WB_1", "Node3_WB_2"],
                        1: ["Node3_EB_0","Node3_EB_1", "Node3_WB_0", "Node3_WB_2"],
                        2: ["Node3_SB_0", "Node3_SB_1", "Node3_SB_2", "Node3_NB_0", "Node3_NB_1", "Node3_NB_2",],
                        3: ["Node3_SB_0", "Node3_SB_2", "Node3_NB_0","Node3_NB_1"],
            },
             exit_detector_map={
                0: ["Node3_NB_1_Exit_1", "Node3_NB_1_Exit_0","Node3_SB_Exit_0","Node3_SB_Exit_1","Node3_EB_Exit_1","Node3_EB_Exit_0","Node3_WB_Exit_0","Node3_WB_Exit_1"],
                1: ["Node3_SB_Exit_0","Node3_SB_Exit_1", "Node3_NB_1_Exit_0","Node3_NB_1_Exit_0"],
                2: ["Node3_WB_Exit_0", "Node3_WB_Exit_1","Node3_SB_Exit_0","Node3_SB_Exit_1","Node3_EB_Exit_1","Node3_EB_Exit_0","Node3_NB_1_Exit_0","Node3_NB_1_Exit_1"],
                3: ["Node3_WB_Exit_0","Node3_WB_Exit_1","Node3_EB_Exit_0","Node3_EB_Exit_1"]
            }
        ),
        QMIXTrafficAgent(
            tls_id="Node4",
            green_phases=[0, 1, 2],
            yellow_phases=[0, 1, 2],
            detectors=[
                "Node4_EB_0", "Node4_EB_1", "Node4_EB_2",
                "Node4_SB_0",
                "Node4_NB_0",
                "Node4_WB_0", "Node4_WB_1", "Node4_WB_2"
            ],
            exit_detectors=[
                "Node4_WB_Exit_0", "Node4_WB_Exit_1", "Node4_NB_Exit_0","Node4_SB_Exit_0",
                "Node4_EB_Exit_0", "Node4_EB_Exit_1"
            ],
            min_green=15,
            max_green=45,
            yellow_duration=6,
            phase_detector_map={
                0: ["Node4_WB_0", "Node4_WB_1", "Node4_WB_2","Node4_EB_0", "Node4_EB_1", "Node4_EB_2"],                          
                1: ["Node4_EB_2"],            
                2: ["Node4_SB_0", "Node4_NB_0"],  
            },
             exit_detector_map={
                0: ["Node4_EB_Exit_0", "Node4_EB_Exit_1","Node4_WB_Exit_0","Node4_WB_Exit_1"],
                1: ["Node4_SB_Exit_0","Node4_WB_Exit_0", "Node4_WB_Exit_1","Node4_NB_Exit_0"],
                2: ["Node4_EB_Exit_0", "Node4_EB_Exit_1","Node4_WB_Exit_0","Node4_WB_Exit_1","Node4_SB_Exit_0","Node4_NB_Exit_0"],
            }
        ),
        QMIXTrafficAgent(
            tls_id="Node5",
            green_phases=[0, 1, 2],
            yellow_phases=[0, 1, 2],
            detectors=[
                "Node5_WB_0", "Node5_WB_1", "Node5_WB_2", "Node5_WB_3", "Node5_WB_4",
                "Node5_SB_0", "Node5_SB_1", "Node5_SB_2", 
                "Node5_EB_0", "Node5_EB_1", "Node5_EB_2", "Node5_EB_3", "Node5_EB_4",
            ],
            exit_detectors=[
                "Node5_SB_Exit_0", "Node5_SB_Exit_1", "Node5_NB_Exit_1_1","Node5_NB_Exit_1_0",
                "Node5_NB_Exit_2_0", "Node5_NB_Exit_2_1","Node5_WB_Exit_0","Node5_WB_Exit_1","Node5_SB_2_Exit_0","Node5_SB_2_Exit_1"
            ],
            min_green=10,
            max_green=35,
            yellow_duration=6,
            phase_detector_map={
                0: ["Node5_WB_0", "Node5_WB_1", "Node5_WB_2", "Node5_WB_3", "Node5_WB_4"],
                1: ["Node5_SB_0", "Node5_SB_1", "Node5_SB_2"], 
                2: ["Node5_EB_0", "Node5_EB_1", "Node5_EB_2", "Node5_EB_3", "Node5_EB_4"],
                
            },
             exit_detector_map={
                0: ["Node5_NB_Exit_1_1","Node5_NB_Exit_1_0", "Node5_NB_Exit_2_0", "Node5_NB_Exit_2_1","Node5_WB_Exit_0","Node5_WB_Exit_1","Node5_SB_Exit_0","Node5_SB_Exit_1"],
                1: ["Node5_NB_Exit_1_1","Node5_NB_Exit_1_0", "Node5_NB_Exit_2_0", "Node5_NB_Exit_2_1","Node5_WB_Exit_0","Node5_WB_Exit_1"],
                2: ["Node5_NB_Exit_1_1","Node5_NB_Exit_1_0", "Node5_NB_Exit_2_0", "Node5_NB_Exit_2_1","Node5_SB_2_Exit_0","Node5_SB_2_Exit_1","Node5_SB_Exit_0","Node5_SB_Exit_1"],
            }
        )
   
    ]

    traci.start(SUMO_CONFIG, stdout=subprocess.DEVNULL)
    run_simulation(agents)
    plot_agent_metrics(agents)
    plot_global_metrics(
        global_metrics["cum_avg_queue"],
        global_metrics["cum_avg_wait"],
        global_metrics["cum_node_throughput"],
        global_metrics["cum_node_co2"]
    )
    traci.close()

if __name__ == "__main__":
    main()