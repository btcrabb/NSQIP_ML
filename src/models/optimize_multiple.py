from src.models import optimization
from src.visualization import visualize_optimization
from src.util import util
from datetime import date

def main():
    models_list = [
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
        out_file = '../../reports/optimization/{}/'.format(model)
        results = '../../reports/optimization/{}/{}_bayes_test.csv'.format(model, today)

        visualize_optimization.main(results=results, out_file=out_file)

    # save all models to disk
    util.save_models('../../reports/optimization/')

if __name__ == "__main__":
    main()