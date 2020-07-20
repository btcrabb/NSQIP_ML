from .base_options import BaseOptions


class DatasetOptions(BaseOptions):
    """This class includes dataset creation options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # raw data options
        parser.add_argument('--input_file', required=True, help='the raw data (should be .csv or .xlsx file')
        parser.add_argument('--output_file', type=str, default = 'data/processed/NSQIP_processed.csv', help='path to save the output file')
        parser.add_argument('--train_test_datasets_path', type=str, default='data/split/',
                            help='path to the directory to save training and testing split datasets')

        # features to include
        parser.add_argument('--feature_list', nargs='+', default = [
            'AGE',
            'SEX',
            'WEIGHT',
            'HEIGHT',
            'RACE_NEW',
            'ETHNICITY_HISPANIC',
            'DIABETES',
            'DYSPNEA',
            'PRSEPIS',
            'ASACLAS',
            'TRANST',
            'DISCHDEST',
            'OPTIME',
            'SMOKE',
            'HXCOPD',
            'HYPERMED',
            'DISCANCR',
            'WNDINF',
            'STEROID',
            'WTLOSS',
            'BLEEDIS',
            'EMERGNCY',
            'RETURNOR',
            'TOTHLOS',
            'PRSODM',
            'PRBUN',
            'PRCREAT',
            'PRALBUM',
            'PRPLATE',
            'PRPTT',
            'PRINR',
            'PRPT',
            'READMISSION1'
        ])
        parser.add_argument('--label', type=str, default='READMISSION1',
                            help='the features to use as the label')

        # processing
        parser.add_argument('--no_bmi', action='store_true', help='Whether to reduce height and weight features into BMI value')
        parser.add_argument('--no_encode_unknown', action='store_true', help='Whether to encode np.NaN values as own category')
        parser.add_argument('--fill_missing', type=str, default='median', help='How to replace missing values in continuous variables [median | mean ]')
        parser.add_argument('--no_normalize', action='store_true', help = 'Whether to normalize continuous data')
        parser.add_argument('--no_train_test_split', action='store_true', help = 'Whether to perform train_test_split' )
        parser.add_argument('--testing_size', type=int, default=300, help = 'size of the testing dataset')

        return parser