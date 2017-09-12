import numpy as np
import base_class

np.random.random(12)


class HyperTS(base_class.Policy):
    def __init__(self, policies):

        self.alpha = {}
        self.beta = {}
        self.policies = policies

    def pullarm(self, arms, user_feature):

        for i in self.policies:
            if i not in self.alpha:
                self.alpha.update({i: 0})
                self.beta.update({i: 0})
        ri = []

        for i in self.policies:
            ri.append(np.random.beta(self.alpha, self.beta))

        policy1 = self.policies[i]
        pulled_arm = policy1.pullarm(user_feature)

        return pulled_arm

    def update_rewards(self, pulled_arm, recommended_arm, reward, policy):

        recommended_arm = int(recommended_arm)
        reward = int(reward)
        pulled_arm = int(pulled_arm)

        if pulled_arm == recommended_arm:
            print 'here'
            if int(reward) == 1:
                self.alpha[policy] += 1
            else:
                self.beta[policy] += 1
            return int(reward)
        else:
            return 0
