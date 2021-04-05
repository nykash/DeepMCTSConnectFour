import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


class C4NN(nn.Module):
    def __init__(self, lr=1e-6):
        super().__init__()
        self.resnet = ResNet18(1, 2)
        self.conv1 = nn.Conv2d(1, 2, (4, 4))
        self.conv2 = nn.Conv2d(2, 3, (2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(18, 9)
        self.fc2 = nn.Linear(9, 9)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.n_actions = 7

    def forward(self, x):
     #   x = self.resnet.forward(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TransitionMemory(object):
    def __init__(self, mem_size, input_shape):
        self.input_shape = input_shape
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size,)+self.input_shape, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,)+self.input_shape, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def reset(self):
        self.state_memory = np.zeros((self.mem_size,) + self.input_shape, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,) + self.input_shape, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.mem_cntr = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        self.mem_cntr %= self.mem_size

    def discount_rewards(self, gamma):
        r_sum = 0
        for i in reversed(range(self.mem_cntr)):
            done = self.terminal_memory[i]
            if (done):
                r_sum = 0

            if(done and i != self.mem_cntr-1):
                break

            r = self.reward_memory[i]
            r_sum = r_sum * gamma + r

            self.reward_memory[i] = r_sum
        T.set_printoptions(precision=4, sci_mode=False)
        np.set_printoptions(precision=4, suppress=True)



class Agent(object):
    def __init__(self, input_shape, gamma, epsilon, lr, batch_size, max_mem_size=100000, eps_end=0.01, eps_dec=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.input_shape = input_shape

        self.p1_mem = TransitionMemory(max_mem_size, input_shape)
        self.p2_mem = TransitionMemory(max_mem_size, input_shape)

        self.policy = C4NN(lr=lr)

        self.action_space = list(range(9))
        self.avg_loss = 0.0
        self.n = 0
        self.avg_losses = []

    def reset_mem(self):
        self.p1_mem.reset()
        self.p2_mem.reset()

    def store_transition_p1(self, state, action, reward, state_, done):
        self.p1_mem.store_transition(state, action, reward, state_, done)

    def store_transition_p2(self, state, action, reward, state_, done):
        self.p2_mem.store_transition(state, action, reward, state_, done)

    def register_win_p1(self, reward):
        self.p2_mem.terminal_memory[self.p2_mem.mem_cntr - 1] = True
        self.p2_mem.reward_memory[self.p2_mem.mem_cntr - 1] = -reward

        self.p1_mem.discount_rewards(self.gamma)
        self.p2_mem.discount_rewards(self.gamma)

    def register_win_p2(self, reward):
        self.p1_mem.terminal_memory[self.p1_mem.mem_cntr - 1] = True
        self.p1_mem.reward_memory[self.p1_mem.mem_cntr - 1] = -reward

        #print(reward, self.p2_mem.reward_memory[self.p2_mem.mem_cntr-1], "rev broda")

        self.p1_mem.discount_rewards(self.gamma)
        self.p2_mem.discount_rewards(self.gamma)

    def choose_action(self, observation, rule=lambda x: True):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation, observation])
            actions = self.policy.forward(state)[0]
          #  print(actions)
            act = actions.argsort()
            k = 6
            while True:
                if (rule(act[k])):
                    break
                k -= 1

            action = act[k]
            print(action, actions, act)
            return action.item()
      #  print("rando bruh")
        actions = []
        for act in self.action_space:
            if (not rule(act)):
                continue
            actions.append(act)
        return np.random.choice(actions)

    def get_learn_params(self, mem):
        max_mem = min(mem.mem_cntr, self.mem_size)
        if(mem.mem_cntr < self.batch_size):
            return None
        batch = np.random.choice(mem.mem_cntr, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(mem.state_memory[batch])
        new_state_batch = T.tensor(mem.new_state_memory[batch])
        reward_batch = T.tensor(mem.reward_memory[batch])
        terminal_batch = T.tensor(mem.terminal_memory[batch])

        action_batch = mem.action_memory[batch]

        policy_eval = self.policy.forward(state_batch)[batch_index, action_batch]
        policy_next = self.policy.forward(new_state_batch)
        policy_next[terminal_batch] = 0.0
        policy_target = reward_batch + self.gamma * T.max(policy_next, dim=1)[0]

        return policy_eval, policy_target

    def learn(self):
        #  print(self.reward_memory, "reward_Mem")
      #  if self.p1_mem.mem_cntr < self.batch_size and self.p2_mem.mem_cntr < self.batch_size:
          #  print("aborting")
           # return

        self.policy.optimizer.zero_grad()


        p1_eval, p1_target = self.get_learn_params(self.p1_mem) if self.p1_mem.mem_cntr > self.batch_size else (None, None)
        p2_eval, p2_target = self.get_learn_params(self.p2_mem) if self.p2_mem.mem_cntr > self.batch_size else (None, None)

        if(p1_eval is None or p2_eval is None):
            policy_eval = p1_eval if p1_eval is not None else p2_eval
        else:
            policy_eval = T.cat([p1_eval, p2_eval])

        if(p1_target is None or p2_target is None):
            policy_target = p1_target if p1_target is not None else p2_target
        else:
            policy_target = T.cat([p1_target, p2_target])

        if(p1_eval is None and p2_eval is None):
            print("aborting")
            return

       # print(policy_eval, policy_target)

      #  policy_eval = p1_eval
      #  policy_target = p1_target

      #  print(p1_eval.shape, p2_eval.shape, policy_eval.shape)

        self.policy.optimizer.zero_grad()

        if(p1_eval is not None):
            loss1 = self.policy.loss(p1_eval, p1_target)
            loss1.backward()

        if(p2_eval is not None):
            loss2 = self.policy.loss(p2_eval, p2_target)
            loss2.backward()
        else:
            loss2 = loss1

        if(p1_eval is None):
            loss1 = loss2


        loss = loss1 + loss2
        self.n += 1
        self.avg_loss = ((self.avg_loss * (self.n-1)) + loss.item())/self.n
        self.avg_losses.append(self.avg_loss)
        print("avg loss is: "+str(self.avg_loss), "reg loss is "+str(loss.item()))

        with open("losses.txt", "a+") as f:
            f.write(str(self.avg_loss)+"\n")

        self.policy.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        print("learned")

    def save(self, path):
        T.save(self.policy, path)

    def load(self, path):
        self.policy = T.load(path)