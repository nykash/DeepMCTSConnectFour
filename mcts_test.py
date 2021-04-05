import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, applications
import numpy as np
from game import ConnectFour
import math

class Agent(object):
    def __init__(self, model):
        self.dataset = [] # board state, policy (actions), color (-1 or 1), z (whether won or lost game), whether z has been calculated or not
        self.model = model

    def predict(self, single_state):
        return np.array(self.model(single_state[np.newaxis, ...]))[:, 0]

    def store_transition(self, state, pi, color):
        self.dataset.append([state, pi, color, 0, False])

    def update_z(self, z):
        # z is terminal from view of player 1
        for i in reversed(range(len(self.dataset))):
            if(self.dataset[i][4]):
                break
            self.dataset[i][3] = z * self.dataset[i][2]
            self.dataset[i][4] = True

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'LeakyReLU': layers.LeakyReLU(0.2)})

    def learn(self):

        states = np.array([self.dataset[i][0] for i in range(len(self.dataset))], dtype=np.float32)
        policies = np.array([self.dataset[i][1] for i in range(len(self.dataset))], dtype=np.float32)
        zs = np.array([self.dataset[i][3] for i in range(len(self.dataset))], dtype=np.float32)

       # print(policies)
        self.model.fit(x=states, y=[policies, zs], epochs=10)

        self.dataset = []

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

        value = -value

        for node in reversed(passed_nodes):
            node.q += value
            node.n += 1

            value = -value



input_boards = layers.Input(shape=(9,))
x_image = layers.Reshape((3, 3, 1))(input_boards)
h_conv4 = layers.Activation('relu')(layers.BatchNormalization(axis=3)(layers.Conv2D(64, 3, padding='valid')(x_image)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
h_conv4_flat = layers.Flatten()(h_conv4)
s_fc1 = layers.Dropout(0.2)(layers.Activation('relu')(layers.BatchNormalization(axis=1)(layers.Dense(50)(h_conv4_flat))))  # batch_size x 1024
s_fc2 = layers.Dropout(0.2)(layers.Activation('relu')(layers.BatchNormalization(axis=1)(layers.Dense(40)(s_fc1))))          # batch_size x 1024
pi = layers.Dense(9, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
v = layers.Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

model = models.Model(inputs=input_boards, outputs=[pi, v])
model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=optimizers.Adam(1e-4))

agent = Agent(model)
agent.load("best_mcts_model.h5")

episodes = 10000

game = ConnectFour()
mcts = MCTS(agent, game)

while True:
    mcts.set_root(game)
    p1_probabilities = mcts.search(1, iters=100)
    p1_action = np.argmax(p1_probabilities)
    print(agent.predict(game.preprocess(1)))
    game.move(p1_action, 1)

    terminal = game.get_terminal(1)
    if terminal is not None:
        break

    game.print_()

    p2_action = int(input("get ya move "))
    game.move(p2_action, -1)

    terminal = game.get_terminal(-1)
    if terminal is not None:
        break

game.print_()



