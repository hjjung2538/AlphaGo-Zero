import argparse
import os
import time
import csv

import gym
import gym_self_go
import Plays
from Plays import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser(description='AlphaGo-Zero Training')
    parser.add_argument('--size', default=5, type=int, choices=[3, 5, 9],
                        help='size of Go (default: 5x5)')
    parser.add_argument('--tau', '--t', default=1, type=int,
                        help='initial infinitesimal temperature (default: 1)')
    parser.add_argument('--search', default=50, type=int,
                        help='number of mcts minibatch search times (default: 10)')
    parser.add_argument('--mb', default=6, type=int,
                        help='minibatch size of mcts (default: 8)')
    parser.add_argument('--vl', default=1, type=int,
                        help='virtual loss (to ensure each thread evaluates different nodes) (default: 1)')
    parser.add_argument('--initial-play', '--iplay', default=2000, type=int,
                        help='number of self play times at initial stage to get play datasets (default: 2000)')
    parser.add_argument('--eval', default=110, type=int,
                        help='number of play times to evaluate neural network (default: 100)')
    parser.add_argument('--train-epochs', default=3, type=int,
                        help='number of train epochs to run (default: 5)')
    parser.add_argument('--tb', default=36, type=int,
                        help='minibatch size of neural network training (default: 20)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', '--m', default=0.9, type=float,
                        help='initial momentum ')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--play', default=250, type=int,
                        help='number of self play times to get more datasets (default: 100)')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total play epochs to run (default: 5)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint', default=50, type=int,
                        help='checkpoint to save (default: 50)')
    args = parser.parse_args()

    #make model
    net_for_train = Net(block=ResidualBlock, blocks=19, size=args.size).cuda()
    net_for_self_play = Net(block=ResidualBlock, blocks=19, size=args.size).cuda()

    #optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            env = gym.make('SelfGo{}x{}-v0'.format(checkpoint['size'],checkpoint['size']))
            args.start_epoch = checkpoint['epoch']
            net_for_self_play.load_state_dict(checkpoint['state_dict'])
            net_for_train.load_state_dict(checkpoint['state_dict2'])
            data_loader = checkpoint['data_loader']
            Plays.dataset = checkpoint['datasets']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:

        #make Go env
        if args.size == 3:
            env = gym.make('SelfGo3x3-v0')
        elif args.size == 5:
            env = gym.make('SelfGo5x5-v0')
        else:
            env = gym.make('SelfGo9x9-v0')

        # make initial play_dataset
        print("Start Initial Play")
        for i in range(args.initial_play):
            env.reset()
            s_self_play = env.state
            play = SelfPlay(state=s_self_play, net=net_for_self_play)
            play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
        print("End Initial Play")
        cat_dataset = torch.utils.data.ConcatDataset(Plays.dataset)
        data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)

    # define loss ft (criterion) and optimizer
    criterion_entropy = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.SGD(net_for_train.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # train - evaluate - self play and get data => how many times?
    for epoch in range(args.start_epoch, args.epochs):
        for train_epoch in range(args.train_epochs):
            for (i, (s, pi, z)) in enumerate(data_loader):
                s = Variable(s).cuda()
                pi = Variable(pi).cuda()
                z = Variable(z).cuda()
                optimizer.zero_grad()
                p_logits, v = net_for_train(s)
                loss = criterion_entropy(p_logits, pi) + criterion_mse(v, z)
                loss.backward()
                optimizer.step()
                print('[%d/%d]. Train_Epoch [%d/%d] Iter [%d] Loss : %.4f' % (
                epoch+1, args.epochs, train_epoch+1, args.train_epochs, i+1, loss.data[0]))
                #csv => write loss
                if train_epoch == 0 and i == 0:
                    if os.path.isfile("Winning_rate_over_training.csv"):
                        with open('Winning_rate_over_training.csv', 'a', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            csvwriter.writerow(['[%d/%d]. Train_Epoch [%d/%d] Iter [%d] Loss : %.4f' % (\
                                                                                                        epoch+1, args.epochs,
                                                                                                        train_epoch+1,
                                                                                                        args.train_epochs, 
                                                                                                        i+1, loss.data[0])])
                    else:
                        with open('Winning_rate_over_training.csv', 'w', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            csvwriter.writerow(['[%d/%d]. Train_Epoch [%d/%d] Iter [%d] Loss : %.4f' % (\
                                                                                                        epoch+1, args.epochs,
                                                                                                        train_epoch+1,
                                                                                                        args.train_epochs, 
                                                                                                        i+1, loss.data[0])]) 
                if train_epoch == (args.train_epochs)-1 and i == len(data_loader)-1:
                        with open('Winning_rate_over_training.csv', 'a', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            csvwriter.writerow(['[%d/%d]. Train_Epoch [%d/%d] Iter [%d] Loss : %.4f' % (\
                                                                                                        epoch+1, args.epochs,
                                                                                                        train_epoch+1,
                                                                                                        args.train_epochs, 
                                                                                                        i+1, loss.data[0])])     

        # evaluator
        rewards = []
        print("Start Evaluate Play")
        for i in range(int(args.eval/2)):
            env.reset()
            s_evaluator = env.state
            evaluator = Evaluator(state=s_evaluator, net_self_player=net_for_self_play, net_train_player=net_for_train)
            rewards.append(evaluator.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl))
            evaluator_v2 = Evaluator(state=s_evaluator, net_self_player=net_for_self_play, net_train_player=net_for_train)
            rewards.append(evaluator_v2.play_v2(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl))
        print("End Evaluate Play")
        print('[{}/{}]. Winning rate : Net train : {}% Net self-play : {}%'.\
              format(epoch+1, args.epochs, 100*rewards.count(1)/len(rewards),100*rewards.count(-1)/len(rewards)))
        with open('Winning_rate_over_training.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['[{}/{}]. Winning rate : Net train : {}% Net self-play : {}%'.\
              format(epoch+1, args.epochs, 100*rewards.count(1)/len(rewards),100*rewards.count(-1)/len(rewards))])
            
        if 100*rewards.count(1)/len(rewards) >= 55:
            net_for_self_play = net_for_train
            Plays.dataset = Plays.dataset[args.play:]

        # get more dataset
        print("Start Self Play to get more datasets")
        for i in range(args.play):
            env.reset()
            s_self_play = env.state
            play = SelfPlay(state=s_self_play, net=net_for_self_play)
            play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
        cat_dataset = torch.utils.data.ConcatDataset(Plays.dataset)
        data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)
        print("End Self Play")

        # save checkpoint
        if (epoch+1) % args.checkpoint == 0:
            torch.save({
                'size': args.size,
                'epoch': epoch+1,
                'state_dict': net_for_self_play.state_dict(),
                'state_dict2': net_for_train.state_dict(),
                'data_loader': data_loader,
                'datasets' : Plays.dataset
                }, 'checkpoint.pth.tar')

    # play with pachi and get winning rate


if __name__ == '__main__':
    main()
