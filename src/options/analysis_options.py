from .base_options import BaseOptions

class AnalysisOptions(BaseOptions):
    """This class includes model analysis options.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # raw data options
        parser.add_argument('--train_test_datasets_path', type=str, default='data/split/',
                            help='The directory of the training/testing datasets'),
        parser.add_argument('--models', type=str, default='models/',
                            help='A directory of models or a single model path (str)')
        parser.add_argument('--bootstrap', action='store_false',
                            help='Do not bootstrap the AUC scores')
        parser.add_argument('--save', action='store_false', default='models/',
                            help='Do not save the ROC curve figures')
        parser.add_argument('--outfile', type=str, default='reports/figures/ROC_AUC_comparison.png',
                            help='The output file for the ROC curves')

        return parser