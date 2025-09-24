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
import tensorflow as tf


GLOBAL_SEED = 41  # Change this to 40,41,42 for a different run

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)


class FixedTimeTrafficAgent:
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
        self.cumulative_co2 = 0

        self.queue_sum_100 = 0.0
        self.queue_count_100 = 0
        self.wait_sum_100 = 0.0
        self.vehicle_count_100 = 0
        self.reward_sum_100 = 0.0

        self.avg_queue_history = []
        self.avg_wait_history = []
        self.cumulative_avg_queue_history = []
        self.cumulative_avg_wait_history = []
        self.throughput_100_history = []
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
    
    def initialize_after_traci(self):
        self.state_size = 2 * len(self.green_phases)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())


    def get_state(self):
        state = []
        for phase in self.green_phases:
            detectors = self.phase_detector_map.get(phase, [])
            
            try:
                q = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)
            except traci.TraCIException:
                q = 0.0

            w = 0.0
            try:
                lanes = [traci.lanearea.getLaneID(d) for d in detectors if traci.lanearea.getLaneID(d)]
                if lanes:
                    w = np.mean([traci.lane.getWaitingTime(l) for l in lanes])
            except traci.TraCIException:
                w = 0.0

            state.append(q / 5.0)
            state.append(w / 10.0)

        # Ensure consistent state size
        while len(state) < self.state_size:
            state.append(0.0)
        if len(state) > self.state_size:
            state = state[:self.state_size]

        return np.array(state, dtype=np.float32)
    
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
        cum_avg_queue = self.total_queue_length_sum / self.queue_measurements if self.queue_measurements else 0
        cum_avg_wait = self.total_vehicle_wait_time / self.vehicle_count if self.vehicle_count else 0
        avg_durations = {
            a: float(np.mean(durs)) if durs else 0.0
            for a, durs in self.predicted_durations.items()
        }

        print(f"[DQN][{self.tls_id}][Step {step}] Avg Queue: {avg_queue:.2f} | Avg Wait: {avg_wait:.2f}s | "
              f"Throughput(100): {self.step_throughput_100} | Total: {self.cumulative_throughput} | Total Co2: {self.cumulative_co2} |"
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


global_metrics = {
    "cum_avg_queue": [],
    "cum_avg_wait": [],
    "cum_node_throughput": [],
    "cum_node_co2": [],
    "cum_network_throughput": [],  
}

def plot_agent_metrics(agents):
    step_interval = 100

    # 1. Avg Queue Plot (All Agents)
    plt.figure(figsize=(10, 6))
    for agent in agents:
        steps = [i * step_interval for i in range(len(agent.avg_queue_history))]
        plt.plot(steps, agent.avg_queue_history, marker='o', label=agent.tls_id)
    plt.title("Fixed Time Average Queue per 100 Steps")
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
    plt.title("Fixed Time Average Wait Time per 100 Steps")
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
    plt.title("Fixed Time Throughput per 100 Steps")
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
    plt.title("Fixed Time Cumulative Average Queue Over Time")
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
    plt.title("Fixed Time Cumulative Average Wait Over Time")
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
    plt.title("Fixed Time Cumulative Throughput Over Time")
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
    plt.title("Fixed Time Cumulative Co2 Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Co2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_global_metrics(cum_avg_queue, cum_avg_wait, cum_node_throughput,cum_node_co2):
    step_interval = 100 
    steps = [i * step_interval for i in range(len(cum_avg_queue))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_avg_queue, label='Fixed Time Cumulative Avg Queue', color='purple')
    plt.title("Fixed Time Global Cumulative Average Queue Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Queue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len( cum_avg_wait))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_avg_wait, label='Fixed Time Cumulative Avg Wait', color='brown')
    plt.title("Fixed Time Global Cumulative Average Wait Time Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Avg Wait (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len(cum_node_throughput))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_node_throughput, label='Fixed Time Cumulative Node Throughput', color='darkgreen')
    plt.title("Fixed Time Global Cumulative Throughput (Node Sum) Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    steps = [i * step_interval for i in range(len(cum_node_co2))]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cum_node_co2, label='Fixed Time Cumulative Node CO2', color='darkgreen')
    plt.title("Fixed Time Global Cumulative Co2 (Node Sum) Over Time")
    plt.xlabel("100-Step Interval")
    plt.ylabel("Total Co2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_simulation(agents, total_steps=500):
    step = 0
    LOG_INTERVAL = 100
    last_step_logged = 0

    global_throughput_100 = 0
    cumulative_global_throughput = 0
    cumulative_global_co2 =0
    queue_window_history, wait_window_history, throughput_window_history = [], [], []

    # === Initialize each agent with fixed green durations ===
    for agent in agents:
        old_logic = traci.trafficlight.getAllProgramLogics(agent.tls_id)[0]
        new_phases = []
        for old_phase in old_logic.phases:
            new_phase = traci.trafficlight.Phase(
                duration=agent.min_green,
                minDur=agent.min_green,
                maxDur=agent.min_green,
                state=old_phase.state
            )
            new_phases.append(new_phase)

        logic = traci.trafficlight.Logic(
            programID="fixed_time",
            type=old_logic.type,
            currentPhaseIndex=old_logic.currentPhaseIndex,
            phases=new_phases
        )
        traci.trafficlight.setProgramLogic(agent.tls_id, logic)

        agent.phase_index = 0
        agent.phase_state = "green"
        agent.phase_timer = agent.min_green
        agent.initialize_after_traci()

        # Ensure these attributes are initialized
        agent.phase_exit_vehicles = set()
        agent.exit_vehicle_seen = set()
        agent.vehicle_wait_times = {}

    while step < total_steps:
        traci.simulationStep()
        step += 1
        arrived = traci.simulation.getArrivedNumber()
        global_throughput_100 += arrived
        cumulative_global_throughput += arrived
        

        for agent in agents:
            idx = agent.phase_index
            phase = agent.green_phases[idx]

            agent.phase_timer -= 1

            if agent.phase_timer <= 0:
                phase_detectors = agent.phase_detector_map.get(idx, [])
                if agent.phase_state == "green":
                    # === Collect metrics before switching phase ===
                    throughput = 0
                    wait_time = 0
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
                    traci.trafficlight.setPhase(agent.tls_id, agent.green_phases[agent.phase_index])
                    agent.phase_state = "green"
                    agent.phase_timer = agent.min_green
                    

            # === Metric collection during GREEN phase ===
            if agent.phase_state == "green":
                # Queue length
                q = sum(traci.lanearea.getLastStepHaltingNumber(d) for d in agent.all_detectors)
                agent.total_queue_length_sum += q
                agent.queue_measurements += 1
                agent.queue_sum_100 += q
                agent.queue_count_100 += 1

                # Record vehicle IDs that passed over exit detectors
                for d in agent.exit_detector_map.get(phase, []):
                    try:
                        vids = traci.lanearea.getLastStepVehicleIDs(d)
                        for vid in vids:
                            agent.phase_exit_vehicles.add(vid)
                    except:
                        continue

                # Update waiting times for all vehicles
                for vid in traci.vehicle.getIDList():
                    try:
                        agent.vehicle_wait_times[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    except:
                        continue

        # === Logging ===
        if step - last_step_logged >= LOG_INTERVAL:
            last_step_logged = step
            window_queue_sum, window_wait_sum, window_throughput = 0, 0, 0

            for agent in agents:
                window_queue_sum += agent.queue_sum_100 / agent.queue_count_100 if agent.queue_count_100 else 0
                window_wait_sum += agent.wait_sum_100 / agent.vehicle_count_100 if agent.vehicle_count_100 else 0
                window_throughput += agent.step_throughput_100
                cumulative_global_co2 += agent.cumulative_co2

                agent.log_metrics()
                agent.print_metrics(step, 0, [0] * agent.action_size)
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

    print("Fixed-time simulation completed.")


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
        FixedTimeTrafficAgent(
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
            min_green=14,
            max_green= 14,
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
        FixedTimeTrafficAgent(
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
            min_green=31,
            max_green=31,
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
        FixedTimeTrafficAgent(
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
            min_green=31,
            max_green=31,
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
        FixedTimeTrafficAgent(
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
            min_green=35,
            max_green=35,
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
        FixedTimeTrafficAgent(
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
            min_green=25,
            max_green=25,
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