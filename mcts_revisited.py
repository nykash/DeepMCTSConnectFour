import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, applications, activations
import numpy as np
from game import ConnectFour
import math
import threading as td

class Agent(object):
    def __init__(self, model):
        self.dataset = [] # board state, policy (actions), color (-1 or 1), z (whether won or lost game), whether z has been calculated or not
        self.guessed_values = []
        self.guessed_actions = []
        self.guessed_states = []
        self.model = model

    def predict(self, single_state):
        return np.array(self.model(single_state[np.newaxis, ...]))[:, 0]

    def store_transition(self, state, pi, color):
        self.dataset.append([state, pi, color, 0, False])

    def store_predicted_val(self, state, val, actions, color):
        self.guessed_states.append(state)
        self.guessed_values.append([val, color])
        self.guessed_actions.append([actions, color])

    def update_z(self, z):
        # z is terminal from view of player 1
        for i in reversed(range(len(self.dataset))):
            if(self.dataset[i][4]):
                break
            self.dataset[i][3] = z * self.dataset[i][2]
            self.dataset[i][4] = True

        for j in reversed(range(len(self.guessed_values))):
            self.guessed_values[j][0] = self.guessed_values[j][1] * z

        for k in reversed(range(len(self.guessed_actions))):
            a = self.guessed_actions[k][0] * self.guessed_actions[k][1] * z
            a = a[np.newaxis, ...]
            self.guessed_actions[k][0] = activations.softmax(a)[0]


    def save(self, path):
        self.model.save(path)


    def learn(self):

        states = np.array([self.dataset[i][0] for i in range(len(self.dataset))], dtype=np.float32)
        policies = np.array([self.dataset[i][1] for i in range(len(self.dataset))], dtype=np.float32)
        zs = np.array([self.dataset[i][3] for i in range(len(self.dataset))], dtype=np.float32)

        states = np.concatenate([states, self.guessed_states], axis=0)
        pis_cat = np.array([self.guessed_values[i][0] for i in range(len(self.guessed_values))])
        zs_cat = np.array([self.guessed_actions[i][0] for i in range(len(self.guessed_actions))])

        policies = np.concatenate([policies, pis_cat], axis=0)
        zs = np.concatenate([zs, zs_cat], axis=0)

       # print(policies)
        self.model.fit(x=states, y=[policies, zs], epochs=1)

        self.dataset = []
        self.guessed_values = []

class Node(object):
    def __init__(self, state, move=None, color=None):
        self.state = state.clone()
        self.move = move

        if(self.move is not None):
            self.state.move(self.move, color)

        self.n = 1
        self.q = 0
        self.pi = 1

        self.children = []

    def get_puct(self, parent_n):
        return (self.q/self.n) + self.pi * math.sqrt(parent_n)/self.n

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, color):
        for move in self.state.get_legal_moves(color):
            self.children.append(Node(self.state, move, color))

    def set_pi(self, pis):
        for child in self.children:
            child.pi = pis[child.move]


class MCTS(object):
    def __init__(self, agent:Agent, game_state):
        self.root = Node(game_state)
        self.agent = agent

    def set_root(self, game_state):
        self.root = Node(game_state)

    def search(self, color, iters=600):
        for i in range(iters):
            self.iter_through(color)

        visits = [0 for i in range(7)]
        for child in self.root.children:
            visits[child.move] = child.n

        return np.array(visits)/sum(visits)

    def iter_through(self, color):
        passed_nodes = [self.root]
        c = color

        while True:
            if(passed_nodes[-1].is_leaf()):
                break

            if(passed_nodes[-1].state.get_terminal(c)):
                break

            max_puct = float("-inf")
            max_puct_index = 0
            for i, child in enumerate(passed_nodes[-1].children):
                puct = child.get_puct(passed_nodes[-1].n)
                if(puct > max_puct):
                    max_puct = puct
                    max_puct_index = i

            passed_nodes.append(passed_nodes[-1].children[max_puct_index])
            c = -c

        leaf = passed_nodes[-1]
        leaf.expand(c)

        pi, value = self.agent.predict(leaf.state.preprocess(color))
        if(leaf.state.get_terminal(-c) is not None):
            value = -leaf.state.get_terminal(-c)

        leaf.set_pi(pi)

        agent.store_predicted_val(leaf.state.preprocess(color), value, pi, color)

        value = -value

        for node in reversed(passed_nodes):
            node.q += value
            node.n += 1

            value = -value




inp = layers.Input(shape=(7, 6, 1))
x = layers.Conv2D(128, (4, 4), padding="same", activation=layers.LeakyReLU(0.2))(inp)
x = layers.Conv2D(64, (3, 3), padding="same", activation=layers.LeakyReLU(0.2))(x)
x = layers.Conv2D(32, (2, 2), padding="same", activation=layers.LeakyReLU(0.2))(x)

pi = layers.Conv2D(32, (2, 2), padding="valid", activation=layers.LeakyReLU(0.2))(x)
pi = layers.Flatten()(pi)
pi = layers.Dropout(0.2)(pi)
pi = layers.BatchNormalization()(pi)
pi = layers.Dense(100, activation="relu")(pi)
pi = layers.Dense(7, activation='softmax', name='pi')(pi)   # batch_size x self.action_size

v = layers.Conv2D(32, (2, 2), padding="valid", activation=layers.LeakyReLU(0.2))(x)
v = layers.Flatten()(v)
v = layers.Dropout(0.2)(v)
v = layers.BatchNormalization()(v)
v = layers.Dense(100, activation="relu")(v)
v = layers.Dense(1, activation='tanh', name='v')(v)                   # batch_size x 1

model = models.Model(inputs=inp, outputs=[pi, v])
model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=optimizers.Adam(1e-4))

class Mem(object):
    def __init__(self):
        self.dataset = []

    def store_transition(self, state, pi, color):
        self.dataset.append([state, pi, color, 0, False])

    def update_z(self, z):
        # z is terminal from view of player 1
        for i in reversed(range(len(self.dataset))):
            if(self.dataset[i][4]):
                break
            self.dataset[i][3] = z * self.dataset[i][2]
            self.dataset[i][4] = True





agent = Agent(model)

episodes = 10000
learn_iters = 1

for i in range(episodes):
    game = ConnectFour()
    mcts = MCTS(agent, game)
    print("episode " + str(i))

    iters = 200
    while True:
        mcts.set_root(game)
        p1_probabilities = mcts.search(1, iters=iters)
        p1_action = np.random.choice(range(7), p=p1_probabilities)

        agent.store_transition(game.preprocess(1), p1_probabilities, 1)
        game.move(p1_action, 1)

        terminal = game.get_terminal(1)
        if terminal is not None:
            agent.update_z(terminal)
            break

        mcts.set_root(game)
        p2_probabilities = mcts.search(-1, iters=iters)
        p2_action = np.random.choice(range(7), p=p2_probabilities)

        agent.store_transition(game.preprocess(-1), p2_probabilities, -1)
        game.move(p2_action, -1)

        terminal = game.get_terminal(-1)
        if terminal is not None:
            agent.update_z(-terminal)
            break

    if(i % 1 == 0):
        agent.learn()
        agent.save("best_mcts_model.h5")


