from neural_nets import *
from game import *
import random

def test():
    model = Agent(input_shape=(3, 3, 3,), gamma=0.99, epsilon=0.0, lr=1e-8, batch_size=64)

    model.load("model.h5")

    wins = 0

    for i in range(100):
        print(i)

        done = False
        game = ConnectFour()
        model.reset_mem()

        while (not done):
            state = game.preprocess(1)
            action = model.choose_action(state, rule=lambda x: x in game.get_legal_moves(1))
            game.move(action, 1)
            reward = game.get_terminal(1)

            if (reward is not None):
                if (reward == 0):
                    print("TIE")
                    done = True
                    continue

                if (reward == 1):
                    game.print_()
                    print("AI WINS")
                    wins += 1
                    done = True
                    continue

            game.print_()

            #move = input("get ya move bruv ")
            move = str(random.randint(0, 7))
            while (not move.isdigit() or (move.isdigit() and int(move) not in game.get_legal_moves(-1))):
               # move = input("pick a move that actually does something -_- ")
                move = str(random.randint(0, 9))

            move = int(move)
            game.move(move, -1)

            reward = game.get_terminal(-1)

            if (reward is not None):
                if (reward == 0):
                    print("TIE")
                    done = True
                    continue

                if (reward == 1):
                    print("HUMAN WINS")
                    done = True
                    continue
           # print(game.preprocess(1), game.preprocess(-1))

    print(wins/100)

def train(episodes=1):
    model = Agent((1, 7, 6), 0.95, 1, 1e-6, 32, eps_dec=1e-3)
 #   model2 = Agent((3, 7, 6), 0.95, 1.0, 1e-6, 32, eps_dec=1e-3)
    for i in range(episodes):
        done = False
        game = ConnectFour()
        if (i % 100 == 0):
            model.reset_mem()
      #  model.reset_mem()
        print("new episode"+str(i)+"\n")
        while (not done):
            state = game.preprocess(1)
            action = model.choose_action(state, rule=lambda x:x in game.get_legal_moves(1))

            game.move(action, 1)

            state_ = game.preprocess(1)
            reward = game.get_terminal(1)
            done = reward is not None
            reward = reward if reward is not None else 0
            model.store_transition_p1(state, action, reward if reward is not None else 0, state_, done)

            if(done):
               # game.print_()
                model.register_win_p1(reward)
                continue

            state = game.preprocess(-1)
            action = model.choose_action(state, rule=lambda x:x in game.get_legal_moves(-1))

            game.move(action, -1)

            state_ = game.preprocess(-1)
            reward = game.get_terminal(-1)
            done = reward is not None
            reward = reward if reward is not None else 0
            model.store_transition_p2(state, action, reward, state_, done)

            if (done):
             #   game.print_()
                model.register_win_p2(reward)
                continue

        #    game.print_()


        model.learn()
        model.save("model.h5")
        #model2.learn()



    #model2.save("model2.h5")

def nani():
    game = ConnectFour()
    model = Agent((3, 7, 6), 0.95, 0.0, 1e-6, 32, eps_dec=1e-3)
    print(model.policy(T.tensor([game.preprocess(1), game.preprocess(1)])))
    game.board = np.ones_like(game.board)
    game.board[np.random.random((7, 6)) > 0.5] = -1
    print(game.board)
    print(model.policy(T.tensor([game.preprocess(1), game.preprocess(1)])))

def main():
    #nani()
    #train(episodes=1000)
    test()


if __name__ == "__main__":
    main()