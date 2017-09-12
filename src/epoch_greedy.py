import numpy as np
import base_class
from sklearn.linear_model import LogisticRegression

np.random.seed(12)


class EpochGreedy(base_class.Policy):
    def __init__(self):
        '''
            arms : List
            Array of arms to choose from
            rewards : List
            Array of rewards on each of the arms
            estimate : List
            Array of present estimated value of each arm
        '''
        self.history = []
        self.label = []

    def pull_arm(self, arms, user_features, exploitation):
        '''
          arms : list
          aricles
          user_feaure : list
          feature vector of user
          num_exploitation : integer
          parameter to decide number of exploration samples we need
        '''
        user_features = map(lambda x: float(x[0]), user_features)
        user_features = np.transpose(user_features)
        arms = map(int, arms)
        if exploitation == 0:
            '''do one step exploration'''
            pulled_arm = np.random.choice(arms)
            self.history.append(user_features)
            self.label.append(pulled_arm)
            return pulled_arm
        else:
            model = LogisticRegression(
                penalty='l2', max_iter=200,
                random_state=12, n_jobs=6,
                multi_class='multinomial',
                solver='lbfgs')

        '''Model should only fit with previous data'''
        model.fit(self.history, self.label)
        prob = model.predict_proba(user_features)
        index_arr = [i for i in model.classes_ if i in arms]
        index_arr = [model.classes_.tolist().index(i) for i in index_arr]
        prob_mod = [prob[0][i] for i in index_arr]
        pulled_arm = self.label[np.argmax(prob_mod)]
        self.history.append(user_features)
        self.label.append(pulled_arm)
        return pulled_arm

    def update_rewards(self, pulled_arm, recommended_arm, reward):
        print pulled_arm, recommended_arm
        pulled_arm = int(pulled_arm)
        recommended_arm = int(recommended_arm)
        if pulled_arm == recommended_arm:
            return int(reward)
        else:
            return -1








