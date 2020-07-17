from src.models import optimization
from src.visualization import visualize_optimization
from datetime import date

def main():
    models_list = [
        'SVC',
        'MLP',
        'XGBoost',
        'DecisionTree',
        'AdaBoost',
        'GradientBoosting'
    ]

    for model in models_list:
        optimization.main(model=model)

        out_file = '../../reports/optimization/{}/'.format(model)
        results = '../../reports/optimization/{}/{}_bayes_test.csv'.format(model, date.today())

        visualize_optimization.main(results=results, out_file=out_file)

if __name__ == "__main__":
    main()