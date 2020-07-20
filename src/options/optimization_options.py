from .base_options import BaseOptions
from datetime import date


class OptimizationOptions(BaseOptions):
    """This class includes dataset creation options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # optimization options
        parser.add_argument('--max_evals', type=int, required=False, default=5000, help='the maximun number of evaluations to run')
        parser.add_argument('--continue_opt', type=int, required=False, default=0, help='continue optimization from previous trials')
        parser.add_argument('--n_folds', type=int, required=False, default=5, help='the number of cross validation folds')
        parser.add_argument('--outfile', type=str, required=False, default='reports/optimization/{}_bayes_test.csv'.format(date.today()), help='the number of cross validation folds')
        parser.add_argument('--metric', type=str, required=False, default='balanced_accuracy_score', help='the metric used to assess cross validation performance')
        parser.add_argument('--model', type=str, required=False, default='SVC', help='the model to optimize [SVC | MLP | XGBoost | DecisionTree | AdaBoost | GradientBoosting]')
        parser.add_argument('--dataset_path', type=str, default='data/split/', help='the path to the training and testing datasets')

        return parser