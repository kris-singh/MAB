import base_class
import numpy as np
import operator
np.random.seed(12)


class LinUCB(base_class.Policy):
    def __init__(self):
        '''
        arms : List
        Array of arms to choose from
        rewards : List
        Array of rewards on each of the arms
        estimate : List
        Array of present estimated value of each arm
        '''
        self.alpha = 0.9
        self.d = 6
        self.Aa = {}
        self.ba = {}
        self.theta_a = {}
        self.pulls = {}

    def pull_arm(self, arms, user_features):
        '''
            arms : list
            aricles
            arms_features : list of list
            feature vector for  arm
            user_feaure : list
            feature vector of user
            num_exploitation : integer
            parameter to decide number of exploration samples we need
        '''
        user_features = map(lambda x: float(x[0]), user_features)
        user_features = np.transpose(user_features)
        for i in arms:
            i = int(i)
            if i in self.pulls.keys():
                '''Not new arm we don't need to intialize'''
                pass
            else:
                self.Aa.update({i: np.identity(6)})
                self.ba.update({i: np.zeros(6)})

            self.theta_a[i] = np.dot(np.linalg.inv(self.Aa[i]), self.ba[i])
            self.pulls[i] = np.dot(self.theta_a[i], user_features) + \
                self.alpha*np.sqrt(np.dot(
                 np.dot(np.transpose(np.array(user_features)),
                        np.linalg.inv(self.Aa[i])), user_features))

        return max(self.pulls.iteritems(), key=operator.itemgetter(1))[0]

    def update_rewards(self, pulled_arm, recommended_arm, reward,
                       user_features):
        noise = 0
        user_features = map(lambda x: float(x[0]), user_features)
        user_features = np.transpose(user_features)
        pulled_arm = int(pulled_arm)
        recommended_arm = int(recommended_arm)
        if pulled_arm == recommended_arm:
            self.Aa[pulled_arm] += np.outer(user_features,
                                            user_features)
            self.ba[pulled_arm] += int(reward) * user_features
            return int(reward) + noise
        else:
            return 0
