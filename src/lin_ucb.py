import base_class
import numpy as np
from sklearn.linearmodel import LogisticRegression
np.seed(12)


def LinUCB(Policy):
    def __init__(self, alpha=0.25):
        '''
        arms : List
        Array of arms to choose from
        rewards : List
        Array of rewards on each of the arms
        estimate : List
        Array of present estimated value of each arm
        '''
        self.alpha = 0.25
        self.d = 6
        self.Aa = {}
        self.ba = {}
        self.theata = {}
        self.pulls = {}

    def pull_arms(self, arms, arms_features, user_features):
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
        for i in arms:
            if i in self.pulls.keys:
                '''Not new arm we don't need to intialize'''
                pass
            else:
                Aa.update(arms: np.identity(36))
                ba.update(arms: np.zeros(36))

            theta_hat = np.dot(np.linalg.inv(Aa[arms]), ba[arms])
            pulls[arms] = (np.dot(np.transpose(theta_hat), user_features) +
                           alpha * np.sqrt(np.dot(np.dot(np.transpose(
                            np.array(user_features)),
                            np.linalg.inv(Aa[arms])), user_features)))
        return pulls_arms.index(max(pulls_arms.values()))

    def update_rewards(self, pulled_arm, recommended_arm, reward):
