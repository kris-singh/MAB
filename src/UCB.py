import base_class
import numpy as np
import math

np.random.seed(12)


class UCB(base_class.Policy):
    def __init__(self):
        '''
        arms : List
        Array of arms to choose from
        rewards : List
        Array of rewards on each of the arms
        estimate : List
        Array of present estimated value of each arm
        '''
        self.estimate = {}
        self.n = {}
        self.tot = 1

    def pull_arm(self, arms):
        '''
        Pull any arm at random from the list of arms
        Update the rewards on the given arm if we select correct arm
        '''
        max1 = 0
        for i in arms:
            i = int(i)
            if i not in self.estimate.keys():
                self.estimate.update({i: 0.1})
                self.n.update({i: 0.1})

            val = self.estimate[i] + math.sqrt(2 * np.log10(self.tot) /
                                               self.n[i])

            if max1 <= val:
                max1 = val
                pulled_arm = i

        self.n[pulled_arm] += 1
        self.tot += 1

        return pulled_arm

    def update_rewards(self, pulled_arm, recommended_arm, reward):
        recommended_arm = int(recommended_arm)
        if pulled_arm == recommended_arm:
            self.estimate[pulled_arm] = (self.n[pulled_arm] -
                                         float(1 / self.n[pulled_arm]) *
                                         self.estimate[pulled_arm] +
                                         (1 / float(self.n[pulled_arm])) *
                                         int(reward))
            return int(reward)

        else:
            return 0
