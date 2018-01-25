import numpy as np
from abc import *
import mcts
from Network import Net
from Network import ResidualBlock
import torch
import torchvision.transforms as transforms

dataset = []

class Play(metaclass=ABCMeta):
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def next_state(self):
        pass

    @abstractmethod
    def play(self):
        pass


class SelfPlay(Play):
    def __init__(self, state, net, parent=None):
        self.state = state
        self.parent = parent
        self.net = net
        self.mcts_node = mcts.Node(state=state, parent=None)
        self.q = -1 if self.state.color == 0 else 1
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.q
        self.mcts_node.channel = self.mcts_node.history() if self.parent == None \
            else np.concatenate((np.concatenate((self.state.board.encode()[:2],
                                                 self.parent.mcts_node.channel))[:16], player_channel))

    def select_action(self, t, iter, batch, virtual_loss):
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net)
        action_num = self.state.board.size ** 2 + 1
        action = np.random.choice(action_num, 1, p=self.pi)[0]
        return action

    def next_state(self, t, iter, batch, virtual_loss):
        child_action = self.select_action(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss)
        child_state = self.state.act(child_action)
        self.child = SelfPlay(state=child_state, net=self.net, parent=self)

    def play(self, t, iter, batch, virtual_loss):
        play_data = []
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss)
            self = self.child
        self.z = self.state.reward()
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net)
        play_data.append((torch.FloatTensor(self.mcts_node.channel), \
                          torch.FloatTensor(self.pi), torch.FloatTensor([self.z])))
        self = self.parent
        while self != None:
            self.z = self.state.reward()
            play_data.append((torch.FloatTensor(self.mcts_node.channel), \
                              torch.FloatTensor(self.pi), torch.FloatTensor([self.z])))
            self = self.parent
        dataset.append(play_data)


class Evaluator(Play):
    def __init__(self, state, net_self_player, net_train_player, parent=None):
        self.state = state
        self.parent = parent
        self.net_self_player = net_self_player
        self.net_train_player = net_train_player
        self.mcts_node = mcts.Node(state=state, parent=None)
        self.q = -1 if self.state.color == 0 else 1
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.q
        self.mcts_node.channel = self.mcts_node.history() if self.parent == None \
            else np.concatenate((np.concatenate((self.state.board.encode()[:2],
                                                 self.parent.mcts_node.channel))[:16], player_channel))

    def select_action(self, t, iter, batch, virtual_loss, net):
        self.pi = self.mcts_node.pi(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        action_num = self.state.board.size ** 2 + 1
        action = np.random.choice(action_num, 1, p=self.pi)[0]
        return action

    def next_state(self, t, iter, batch, virtual_loss, net):
        child_action = self.select_action(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=net)
        child_state = self.state.act(child_action)
        self.child = Evaluator(state=child_state, net_self_player=self.net_self_player,\
                               net_train_player=self.net_train_player, parent=self)

    def play(self, t, iter, batch, virtual_loss):
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_self_player)
            self = self.child
            if self.state.board.is_terminal != True:
                self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_train_player)
                self = self.child
            else:
                break
        return self.state.reward()

    def play_v2(self, t, iter, batch, virtual_loss):
        while self.state.board.is_terminal != True:
            self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_train_player)
            self = self.child
            if self.state.board.is_terminal != True:
                self.next_state(t=t, iter=iter, batch=batch, virtual_loss=virtual_loss, net=self.net_self_player)
                self = self.child
            else:
                break
        return self.state.reward_v2()



