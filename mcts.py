import gym
import gym_self_go
import numpy as np
from math import sqrt
from collections import defaultdict
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = defaultdict(lambda: None)
        action_num = state.board.size ** 2 + 1
        self.N = np.zeros([action_num], dtype=np.float32)
        self.W = np.zeros([action_num], dtype=np.float32)
        self.p = np.zeros([action_num], dtype=np.float32)
        self.q = -1 if self.state.color-1 == 0 else 1
        self.searched = 0

    def select(self):
        c = 1
        self.U = c * (sqrt(sum(self.N))) / (1 + self.N) * self.p
        self.Q = np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N != 0)

    def expand(self, child_state, action):
        child = Node(state=child_state, parent=self)
        self.children[action] = child
        self.children[action].action = action

    def history(self):
        player_channel = np.ones((1, self.state.board.size, self.state.board.size)) * self.q
        if self.parent == None:
            initial = np.concatenate([np.zeros((2, self.state.board.size, self.state.board.size)).\
                                     astype(int) for _ in range(7)])
            history = np.concatenate((self.state.board.encode()[:2], initial))[:16]
            self.channel = np.concatenate((history, player_channel))
        else:
            self.channel = np.concatenate((self.state.board.encode()[:2], self.parent.channel))[:16]
            self.channel = np.concatenate((self.channel, player_channel))
        return self.channel

    def search(self, virtual_loss):
        if self.searched == 0:
            leaf = self
            n_input = self.channel
            self.searched += 1
        else:
            # select
            self.select()
            score = (self.U + self.Q * self.q) + self.state.illegal_actions_to_minus()
            action = randargmax(score)
            self.N[[action]] += virtual_loss
            self.W[[action]] += virtual_loss * self.q * -1
            while self.children[action] != None:
                self = self.children[action]
                self.select()
                score = (self.U + self.Q * self.q) + self.state.illegal_actions_to_minus()
                action = randargmax(score)
                self.N[[action]] += virtual_loss
                self.W[[action]] += virtual_loss * self.q * -1
                # expand
            if self.state.board.is_terminal == False:
                new_state = self.state.act(action)
                self.expand(new_state, action)
                self = self.children[action]
                leaf = self
                n_input = self.history()
            else:
                ##if node is terminal => return or not?
                leaf = self
                n_input = self.channel
        return leaf, n_input

    def eval_and_backup(self, batch, virtual_loss, net):
        leaves = []
        n_inputs = []

        # 8 mini batch
        for i in range(batch):
            leaf, n_input = self.search(virtual_loss)
            leaves.append(leaf)
            n_inputs.append(torch.FloatTensor(n_input))

        n_inputs = Variable(torch.stack(n_inputs)).cuda()
        #eval
        p, v = net(n_inputs)
        p = F.softmax(p)
        p = p.data.cpu().numpy()
        v = v.data.cpu().numpy()
        for i in range(batch):
            self = leaves[i]
            self.p = p[i]*0.75 + np.random.dirichlet(np.ones([self.state.board.size**2+1])*0.03)*0.25
            # backup
            while self.parent != None:
                self.parent.N[[self.action]] += 1
                self.parent.W[[self.action]] += v[i]
                self = self.parent

    def pi(self, t, iter, batch, virtual_loss, net):
        action_num = self.state.board.size ** 2 + 1
        pi = np.zeros([action_num], dtype=np.float32)
        for i in range(iter):
            self.eval_and_backup(batch, virtual_loss, net)
        for i in range(action_num):
            pi[[i]] = ((self.N[[i]])**(1/t))/((sum(self.N))**(1/t))
        return pi


def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)