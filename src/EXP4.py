import numpy as np
import base_class
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

np.random.random(12)


class EXP4(base_class.Policy):
    def __init__(self, eta, history, k):
        self.eta = eta
        self.data = history
        self.k = k
        self.estimate = {}
        self.probs = {}
        self.expert = {}
        self.experts = 0

    def func(self):
        self.experts = self.get_expert()

    def train_expert(self, expert):
        ctx = []
        label = []
        with open(self.data, 'r') as f:
            data = f.read()
        data = data.split('\n')
        for i in data:
            line = i.split('|')
            article_yahoo_recommended = line[0].split(' ')[1]
            article_yahoo_reward = line[0].split(' ')[2]
            feature_vec = map(lambda x: x.split(':'), line[1].split(' ')[1:])
            feature_vec = np.array([j[1] for j in feature_vec[:-1]])
            self.user = feature_vec  # current user feature vector
            self.user = list(map(float, self.user))
            if int(article_yahoo_reward) == 1:
                ctx.append(self.user)
                label.append(int(article_yahoo_recommended))

        expert.fit(ctx, label)
        return expert

    def get_expert(self):
        logreg = OneVsRestClassifier(LogisticRegression())
        # mnb = OneVsRestClassifier(MultinomialNB())
        expert1 = self.train_expert(logreg)
        return expert1

    def pull_arm(self, arms, user_features):
        '''
            Pull any arm at random from the list of arms
            Update the rewards on the given arm if we select correct arm
        '''
        experts = self.experts
        arms = [int(i) for i in arms]
        user_features = list(map(float, self.user))
        self.prob = {}
        for i in arms:
            '''Update new arms with intial weigt 1'''
            if i in self.estimate:
                    pass
            else:
                self.estimate.update({i: 1})
        pred_proba = experts.predict_proba(user_features)
        for k, i in enumerate(arms):
            val = 0
            sum_weight = sum(self.estimate.values())
            val += ((self.estimate[i] * pred_proba[0][k]) / sum_weight)
            self.prob.update({i: (1 - self.eta) * val + 0.1})

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
            print 'here'
            self.estimate[pulled_arm] *= np.exp(
                    self.eta * (reward / (self.prob[pulled_arm] *
                                          len(self.estimate))))
            return int(reward)
        else:
            return 0
