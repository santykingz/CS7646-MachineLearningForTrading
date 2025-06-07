import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learner = bl.BagLearner(
            learner = bl.BagLearner,
            kwargs = {'learner': lrl.LinRegLearner, 'kwargs': {}, 'bags': 20,
                      'boost': False,'verbose': verbose},
            bags = 20, boost = False, verbose = verbose)
    def author(self):
        return "sspickard3"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        self.learner.add_evidence(data_x, data_y)
    def query(self, points):
        res = self.learner.query(points)
        return res