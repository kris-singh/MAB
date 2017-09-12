import numpy as np
from RandomPolicy import RandomPolicy
from LinUCB import LinUCB
from epsilon_greedy import Epsilon_Greedy
from UCB import UCB
from EXP3 import EXP3
from EXP4 import EXP4
import matplotlib.pyplot as plt
import warnings


class OfflinePolicyEvaluator():
    def __init__(self, policy, logged_data):
        '''
        policy : MAB policy to evaluate
        logged_data : data that contains the information
        '''
        self.policy = policy
        self.data = logged_data
        self.article_clicked = {}
        self.user = np.zeros([6, 1])
        self.article = {}

    def read_file(self):
        policy1 = RandomPolicy()
        policy2 = LinUCB()
        policy = Epsilon_Greedy()
        policy3 = UCB()
        policy4 = EXP3(0.3)
        policy5 = EXP4(0.3, './data/fivepercent.txt', 2)
        policy5.func()
        # policy6 = HyperTS(policy,policy4)
        eps_1 = []
        eps_2 = []
        eps_3 = []
        cum_reg = []
        cum_reg1 = []
        cum_reg2 = []
        cum_reg3 = []
        cum_reg4 = []
        cum_reg5 = []
        cum_reg6 = []
        with open(self.data, 'r') as f:
            data = f.read()

        data = data.split('\n')
        for i in data:
            line = i.split('|')
            article_yahoo_recommended = line[0].split(' ')[1]
            article_yahoo_reward = line[0].split(' ')[2]
            feature_vec = map(lambda x: x.split(':'), line[1].split(' ')[1:])
            feature_vec = np.array([j[1] for j in feature_vec[:-1]])
            feature_vec = feature_vec.reshape((6, 1))
            self.user = feature_vec  # current user feature vector
            self.article = {}
            for k in line[2:len(line)-1]:
                '''Calculating the article feature vectors'''
                feature_vec = map(lambda x: x.split(':'), k.split(' ')[1:])
                feature_vec = np.array([j[1] for j in feature_vec[:-1]])
                if feature_vec.shape[0] != 6:
                    '''skip because of 109528 its length vector is 1'''
                    continue
                feature_vec = feature_vec.reshape((6, 1))
                self.article[k.split(' ')[0]] = feature_vec

            '''
               computing for last article as some issues with splitting
               last element has no [[]] element in the split
               easy solution refine if time avails
            '''
            feature_vec = map(lambda x: x.split(':'), line[-1].split(' ')[1:])
            feature_vec = np.array([j[1] for j in feature_vec])
            if feature_vec.shape[0] != 6:
                '''skip because of 109528 its length vector is 1'''
                continue

            feature_vec = feature_vec.reshape((6, 1))
            self.article[line[-1].split(' ')[0]] = feature_vec
            size = len(self.article.keys())
            arms = np.array(self.article.keys())
            arms_features = np.array(self.article.values())
            user_features = np.array(self.user)
            '''
            #Random
            pulled_arm1 = policy1.pull_arm(arms,arms_features,user_features)
            ret_val1 = policy1.update_rewards(pulled_arm1,
                article_yahoo_recommended,article_yahoo_reward)
            if len(cum_reg1) == 0:
                cum_reg1.append(ret_val1)
            else:
                cum_reg1.append(cum_reg1[-1] + ret_val1)
            #LinUCB
            pulled_arm2 = policy2.pull_arm(arms,user_features)
            ret_val2 = policy2.update_rewards(pulled_arm2,article_yahoo_recommended,
                article_yahoo_reward,user_features)
            if len(cum_reg2) == 0:
                cum_reg2.append(ret_val2)
            else:
                cum_reg2.append(cum_reg2[-1] + ret_val2)
            #Epsilon-Greedy
            pulled_arm = policy.pull_arm(arms,0.3)
            ret_val = policy.update_rewards(pulled_arm,article_yahoo_recommended,
                        article_yahoo_reward)
            if len(cum_reg) == 0:
                cum_reg.append(0)
            else:
                cum_reg.append(cum_reg[-1] + ret_val)
            '''
            # UCB
            pulled_arm3 = policy3.pull_arm(arms)
            ret_val3 = policy3.update_rewards(pulled_arm3,
                                              article_yahoo_recommended,
                                              article_yahoo_reward)
            if len(cum_reg3) == 0:
                cum_reg3.append(ret_val3)
            else:
                cum_reg3.append(cum_reg3[-1] + ret_val3)
            # exp3
            pulled_arm4 = policy4.pull_arm(arms)

            ret_val4 = policy4.update_rewards(pulled_arm4,
                                              article_yahoo_recommended,
                                              article_yahoo_reward)

            if len(cum_reg4) == 0:
                cum_reg4.append(ret_val4)
            else:
                cum_reg4.append(cum_reg4[-1] + ret_val4)

            pulled_arm5 = policy5.pull_arm(arms, user_features)
            ret_val5 = policy5.update_rewards(pulled_arm5,
                                              article_yahoo_recommended,
                                              article_yahoo_reward)
            if len(cum_reg5) == 0:
                cum_reg5.append(ret_val5)
            else:
                cum_reg5.append(cum_reg5[-1] + ret_val5)
            # pulled_arm6 = policy6.pull_arm()
        # plt.plot(np.arange(len(cum_reg)),cum_reg,'g')
        # plt.plot(np.arange(len(cum_reg1)),cum_reg1,'b')
        # plt.plot(np.arange(len(cum_reg2)),cum_reg2,'y')
        plt.plot(np.arange(len(cum_reg3)), cum_reg3, 'm')
        # plt.plot(np.arange(len(cum_reg4)),cum_reg4,'k')
        plt.plot(np.arange(len(cum_reg5)), cum_reg5, 'b')
        plt.legend(['LRU', 'CENC'])
        plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test = OfflinePolicyEvaluator(None, './data/fivepercent.txt')
    test.read_file()
