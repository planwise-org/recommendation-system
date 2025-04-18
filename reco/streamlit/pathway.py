import torch
import torch.nn as nn
import torch.optim as optim
import random
import networkx as nx
import numpy as np
from collections import deque
import math

# Hyperparameters
GAMMA = 0.9
EPSILON = 0.2
LR = 0.01
NUM_EPISODES = 300
MAX_STEPS = 20

# Reward weights
W_PREF = 0.8
W_DIST = 0.3
W_DIVERSITY = 0.6
W_RATING = 0.3 

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class RLPathPlanner:
    def __init__(self, places, user_lat, user_lng):
        self.places = places
        self.graph = self._build_graph()
        self.num_nodes = len(places)
        self.model = QNetwork(input_size=self.num_nodes + 1, hidden_size=64, output_size=self.num_nodes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.user_start = (user_lat, user_lng)

    def _build_graph(self):
        G = nx.Graph()
        for i, place in enumerate(self.places):
            if 'category' not in place:
                place['category'] = 'misc'  # <-- ensure it's set
            G.add_node(i, **place)
        
        for i in G.nodes:
            for j in G.nodes:
                if i != j:
                    dist = self.haversine(G, i, j)
                    G.add_edge(i, j, weight=dist)
        
        return G

    def haversine(self, G, i, j):
        lat1, lon1 = G.nodes[i]['lat'], G.nodes[i]['lng']
        lat2, lon2 = G.nodes[j]['lat'], G.nodes[j]['lng']
        R = 6371
        phi1, phi2 = map(math.radians, [lat1, lat2])
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    
    def get_state(self, current, visited):
        visited_vector = np.array([1 if i in visited else 0 for i in range(self.num_nodes)])
        return torch.FloatTensor(np.append(visited_vector, current))

    def compute_reward(self, current, next_node, visited_categories):
        # Distance calculation
        if current == -1:
            dist = 0  # No distance penalty for the first move
        else:
            try:
                dist = self.graph[current][next_node].get('weight', 0)
            except KeyError:
                dist = 0  # Fallback if edge is missing (shouldn’t happen after fix)

        # Preference score
        preference_score = self.graph.nodes[next_node].get('score', 0)

        # Diversity bonus
        category = self.graph.nodes[next_node].get('category', 'misc')
        diversity_bonus = 1.0 if category and category not in visited_categories else 0.0

        # Ratings and reviews
        rating = self.graph.nodes[next_node].get('actual_rating', 0)
        num_reviews = self.graph.nodes[next_node].get('user_ratings_total', 0)

        rating_norm = rating / 5.0
        max_reviews = max(
            [self.graph.nodes[n].get('user_ratings_total', 0) for n in self.graph.nodes],
            default=1
        )
        reviews_norm = np.log(num_reviews + 1) / np.log(max_reviews + 1)
        rating_score = 0.7 * rating_norm + 0.3 * reviews_norm

        # Final reward
        reward = (
            W_PREF * preference_score
            - W_DIST * dist
            + W_DIVERSITY * diversity_bonus
            + W_RATING * rating_score
        )

        return reward

    def train(self):
        for episode in range(NUM_EPISODES):
            visited = set()
            visited_categories = set()
            path = []
            current = -1  # -1 = start (not a node yet)
            total_reward = 0

            for step in range(MAX_STEPS):
                state = self.get_state(current, visited)
                q_values = self.model(state)

                if random.random() < EPSILON:
                    action = random.choice([i for i in range(self.num_nodes) if i not in visited])
                else:
                    mask = torch.tensor([float('-inf') if i in visited else 0 for i in range(self.num_nodes)])
                    action = torch.argmax(q_values + mask).item()

                reward = self.compute_reward(current if current != -1 else action, action, visited_categories)
                total_reward += reward

                next_state = self.get_state(action, visited.union({action}))
                next_q_values = self.model(next_state)
                max_next_q = torch.max(next_q_values).detach()

                target_q = q_values.clone()
                target_q[action] = reward + GAMMA * max_next_q

                loss = self.criterion(q_values, target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                visited.add(action)
                visited_categories.add(self.graph.nodes[action].get('category', 'misc'))
                path.append(action)
                current = action

                if len(visited) == self.num_nodes:
                    break

            if episode % 50 == 0:
                print(f"Episode {episode} | Total reward: {total_reward:.2f} | Path length: {len(path)}")

        print("Training complete.")

    def get_best_path(self, start_lat, start_lng):
        visited = set()
        visited_categories = set()
        path = []
        current = -1

        for _ in range(MAX_STEPS):
            state = self.get_state(current, visited)
            q_values = self.model(state)

            mask = torch.tensor([float('-inf') if i in visited else 0 for i in range(self.num_nodes)])
            action = torch.argmax(q_values + mask).item()

            if action in visited:
                break

            visited.add(action)
            visited_categories.add(self.graph.nodes[action]['category'])
            path.append(action)
            current = action
        
        return [self.places[i] for i in path]
    
def get_optimal_path(recommendations, user_lat, user_lng, train=True):
    planner = RLPathPlanner(recommendations, user_lat, user_lng)
    if train:
        planner.train()
    return planner.get_best_path(user_lat, user_lng)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # in km

def reorder_with_tsp(places):
    """Reorders a list of places using TSP and Haversine distance."""
    import networkx as nx
    G = nx.complete_graph(len(places))

    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            dist = haversine_distance(
                places[i]['lat'], places[i]['lng'],
                places[j]['lat'], places[j]['lng']
            )
            G[i][j]['weight'] = dist
            G[j][i]['weight'] = dist  # since it's undirected

    tsp_order = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=False)

    return [places[i] for i in tsp_order]