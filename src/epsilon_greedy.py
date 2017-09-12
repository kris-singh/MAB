import base_class
import numpy as np

np.random.seed(12)


class Epsilon_Greedy(base_class.Policy):
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

    def pull_arm(self, arms, epsilon):
        '''
            Pull any arm at random from the list of arms
            Update the rewards on the given arm if we select correct arm
            
        '''
        max_val = 0
        max_key = 0
        if epsilon > round(np.random.random(), 2):
            '''exploit'''
            for i in arms:
                i = int(i)
                if i in self.estimate.keys():
                    if self.estimate[i] > max_val:
                        max_val = self.estimate[i]
                        max_key = i
                else:
                    self.estimate.update({i: 0})
                    if max_val == 0:
                        max_key = i
            pulled_arm = max_key

        else:
            '''explore'''
            pulled_arm = np.random.choice(arms)

        return int(pulled_arm)

    def update_rewards(self, pulled_arm, recommended_arm, reward):
        noise = np.random.normal(0, 1)
        noise = 0
        recommended_arm = int(recommended_arm)
        if pulled_arm == recommended_arm:
            if pulled_arm in self.estimate:
                self.estimate[pulled_arm] += int(reward) + noise
            else:
                self.estimate.update({pulled_arm: int(reward) + noise})
            return int(reward) + noise

        else:
            return 0
