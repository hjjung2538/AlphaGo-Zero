import argparse
import os
import time
import csv
from tqdm import tqdm
from copy import copy

import gym
import gym_self_go
from Plays import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms



def main():
    datasets = []
    
    parser = argparse.ArgumentParser(description='AlphaGo-Zero Training')
    parser.add_argument('--size', default=5, type=int, choices=[3, 5, 9],
                        help='size of Go (default: 9x9)')
    parser.add_argument('--tau', '--t', default=1, type=int,
                        help='initial infinitesimal temperature')
    parser.add_argument('--search', default=85, type=int,
                        help='number of mcts minibatch search times')
    parser.add_argument('--mb', default=5, type=int,
                        help='minibatch size of mcts')
    parser.add_argument('--vl', default=1, type=int,
                        help='virtual loss (to ensure each thread evaluates different nodes)')
    parser.add_argument('--initial-play', '--iplay', default=500, type=int,
                        help='number of self play times at initial stage to get play datasets')
    parser.add_argument('--eval', default=100, type=int,
                        help='number of play times to evaluate neural network')
    parser.add_argument('--train-epochs', default=6, type=int,
                        help='number of train epochs to run')
    parser.add_argument('--tb', default=36, type=int,
                        help='minibatch size of neural network training')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', '--m', default=0.9, type=float,
                        help='initial momentum ')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--play', default=250, type=int,
                        help='number of self play times to get more datasets')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total play epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint', default=2, type=int,
                        help='checkpoint to save')
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
            net_for_self_play.load_state_dict(checkpoint['state_dict_self_play'])
            net_for_train.load_state_dict(checkpoint['state_dict_train'])
            data_loader = checkpoint['data_loader']
            datasets = checkpoint['datasets']
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
        for i in tqdm(range(args.initial_play)):
            env.reset()
            s_self_play = env.state
            play = SelfPlay(state=s_self_play, net=net_for_self_play)
            play_data = play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
            datasets.append(dihedral_transformation(play_data))
        print("End Initial Play")
        cat_dataset = torch.utils.data.ConcatDataset(datasets)
        data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)

    # define loss ft (criterion) and optimizer
    criterion_entropy = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.SGD(net_for_train.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # train - evaluate - self play and get more data 
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
        for i in tqdm(range(int(args.eval/2))):
            env.reset()
            s_evaluator = env.state
            evaluator = Evaluator(state=s_evaluator, net_self_player=net_for_self_play, net_train_player=net_for_train)
            rewards.append(evaluator.play_train_net_white(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl))
            rewards.append(evaluator.play_train_net_black(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl))
        print("End Evaluate Play")
        print('[{}/{}]. Winning rate : Net train : {}% Net self-play : {}%'.\
              format(epoch+1, args.epochs, 100*rewards.count(1)/len(rewards),100*rewards.count(-1)/len(rewards)))
        with open('Winning_rate_over_training.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['[{}/{}]. Winning rate : Net train : {}% Net self-play : {}%'.\
              format(epoch+1, args.epochs, 100*rewards.count(1)/len(rewards),100*rewards.count(-1)/len(rewards))])
            
        if 100*rewards.count(1)/len(rewards) >= 100*rewards.count(-1)/len(rewards):
            net_for_self_play = net_for_train
            datasets = datasets[args.play:]

        # get more dataset
        print("Start Self Play to get more datasets")
        for i in tqdm(range(args.play)):
            env.reset()
            s_self_play = env.state
            play = SelfPlay(state=s_self_play, net=net_for_self_play)
            play_data = play.play(t=args.tau, iter=args.search, batch=args.mb, virtual_loss=args.vl)
            datasets.append(dihedral_transformation(play_data))
        cat_dataset = torch.utils.data.ConcatDataset(datasets)
        data_loader = torch.utils.data.DataLoader(cat_dataset, batch_size=args.tb)
        print("End Self Play")

        # save checkpoint
        if (epoch+1) % args.checkpoint == 0:
            torch.save({
                'size': args.size,
                'epoch': epoch+1,
                'state_dict_self_play': net_for_self_play.state_dict(),
                'state_dict_train': net_for_train.state_dict(),
                'data_loader': data_loader,
                'datasets' : datasets
                }, 'checkpoint.pth.tar')

    # play with pachi and get winning rate


def rotate_square_matrix_right_90(matrix):
    """Rotate an NxN matrix 90 degrees clockwise."""
    n = len(matrix)
    for layer in range((n + 1) // 2):
        for index in range(layer, n - 1 - layer, 1):
            matrix[layer][index], matrix[n - 1 - index][layer], \
                matrix[index][n - 1 - layer], matrix[n - 1 - layer][n - 1 - index] = \
                matrix[n - 1 - index][layer], matrix[n - 1 - layer][n - 1 - index], \
                matrix[layer][index], matrix[index][n - 1 - layer]
    return matrix    
    
    
def dihedral_transformation(play_data):
    """Rotate(90, 180, 270 degrees clockwise) and flip Go board(data augmentation)"""
    play_data_list = []
    for i, (s, pi, z) in enumerate(play_data):
        data_aug = []
        for l in range(8):
            data_aug.append([])
        #rotate, reflect s
        for j in s:
            #rotation
            data_aug[0].append(j)
            j2 = copy(j)
            data_aug[1].append(rotate_square_matrix_right_90(j2))
            j3 = copy(j2)
            data_aug[2].append(rotate_square_matrix_right_90(j3))
            j4 = copy(j3)
            data_aug[3].append(rotate_square_matrix_right_90(j4))
            #reflection
            j5 = copy(j)
            data_aug[4].append(np.fliplr(j5))
            j6 = copy(data_aug[4][-1])
            data_aug[5].append(rotate_square_matrix_right_90(j6))
            j7 = copy(j6)
            data_aug[6].append(rotate_square_matrix_right_90(j7))
            j8 = copy(j7)
            data_aug[7].append(rotate_square_matrix_right_90(j8))
        #rotate, reflect pi
        #rotation
        data_aug[0].append(pi)
        pi2 = copy(pi)
        p2 = rotate_square_matrix_right_90(pi2[:25].reshape(5,5))
        data_aug[1].append(np.append(p2.reshape(1,25), pi[25:])) 
        pi3 = copy(p2)
        p3 = rotate_square_matrix_right_90(pi3)                                    
        data_aug[2].append(np.append(p3.reshape(1,25), pi[25:]))
        pi4 = copy(p3)
        p4 = rotate_square_matrix_right_90(pi4)
        data_aug[3].append(np.append(p4.reshape(1,25), pi[25:]))
        #reflection
        pi5 = copy(pi)   
        p5 = np.fliplr(pi[:25].reshape(5,5))
        data_aug[4].append(np.append(p5.reshape(1,25), pi[25:])) 
        pi6 = copy(p5)
        p6 = rotate_square_matrix_right_90(pi6)                                    
        data_aug[5].append(np.append(p6.reshape(1,25), pi[25:]))
        pi7 = copy(p6)
        p7 = rotate_square_matrix_right_90(pi7)                                    
        data_aug[6].append(np.append(p7.reshape(1,25), pi[25:]))
        pi8 = copy(p7)
        p8 = rotate_square_matrix_right_90(pi8)                                    
        data_aug[7].append(np.append(p8.reshape(1,25), pi[25:]))
        for data in data_aug:
            play_data_list.append((torch.FloatTensor(np.concatenate(data[:17]).reshape(17,5,5)), 
                              torch.FloatTensor(data[17]), torch.FloatTensor([z])))
    return play_data_list

                                  
if __name__ == '__main__':
    main()
