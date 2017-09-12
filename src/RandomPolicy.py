import base_class
import numpy as np

np.random.seed(12)


class RandomPolicy(base_class.Policy):
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

    def pull_arm(self, arms, arms_features, user_features):
        '''
        Pull any arm at random from the list of arms
        Update the rewards on the given arm if we select correct arm
        '''
        pulled_arm = np.random.choice(arms)
        return pulled_arm

    def update_rewards(self, pulled_arm, recommended_arm, reward):
        noise = np.random.normal(0, 1)
        noise = 0
        print pulled_arm, recommended_arm
        if pulled_arm == recommended_arm:
            if pulled_arm in self.estimate:
                self.estimate[pulled_arm] += int(reward) + noise
            else:
                self.estimate.update({pulled_arm: int(reward) + noise})
            return int(reward) + noise
        else:
            return 0 + noise
