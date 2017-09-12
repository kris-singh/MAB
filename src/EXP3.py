import base_class
import numpy as np

np.random.seed(12)


class EXP3(base_class.Policy):
    def __init__(self, eta):
        '''
        arms : List
        Array of arms to choose from
        rewards : List
        Array of rewards on each of the arms
        estimate : List
        Array of present estimated value of each arm
        '''
        self.estimate = {}
        self.eta = eta
        self.probs = {}

    def pull_arm(self, arms):
        '''
        Pull any arm at random from the list of arms
        Update the rewards on the given arm if we select correct arm
        '''
        arms = [int(i) for i in arms]
        for i in arms:
            '''Update new arms with intial weigt 1'''
            if i in self.estimate:
                pass
            else:
                self.estimate.update({i: 1})
        sum_weight = sum(self.estimate.values())
        self.prob = {}
        for i in arms:
            self.prob.update({i: (1-self.eta) *
                             self.estimate[i] / sum_weight +
                             self.eta * len(self.estimate)})

        z = np.random.random()
        cum_prob = 0.0
        for k, v in self.prob.items():
            cum_prob += v
            if cum_prob > z:
                return k

    def update_rewards(self, pulled_arm, recommended_arm, reward):

        recommended_arm = int(recommended_arm)
        reward = int(reward)
        pulled_arm = int(pulled_arm)
        if pulled_arm == recommended_arm:
            self.estimate[pulled_arm] *= np.exp(self.eta *
                                                (reward /
                                                 (self.prob[pulled_arm] *
                                                  len(self.estimate))))
            return int(reward)

        else:
            return 0
