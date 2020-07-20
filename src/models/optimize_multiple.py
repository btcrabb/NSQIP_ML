import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import optimization
from visualization import visualize_optimization
from util import util
from datetime import date

def main():
    models_list = [
        'SVC',
        'MLP',
        'XGBoost',
        'DecisionTree',
        'AdaBoost',
        'RandomForest',
        'GradientBoosting'
    ]

    for model in models_list:
        today = date.today()

        #  run main optimization
        optimization.main(model=model)

        # visualize the results
        out_file = 'reports/optimization/{}/'.format(model)
        results = 'reports/optimization/{}/{}_bayes_test.csv'.format(model, today)

        visualize_optimization.main(results=results, out_file=out_file)

    # save all models to disk
    util.save_models('models/')

if __name__ == "__main__":
    main()