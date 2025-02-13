import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument("--exp_name", type=str, default='Run-SWGCN', help="Message for run.")
    parser.add_argument('--model', type=str, default='SWGCN', help='Choose a model from {BPRMF, SWGCN}.')
    parser.add_argument('--model_path', type=str, default='logs/', help='Store model path.')
    parser.add_argument('--load_model', type=int, default=0, help='Weather to load model')
    parser.add_argument('--gpu_id', type=int, default=0, help='Choose a gpu id')
    parser.add_argument('--n_worker', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--save_log', type=int, default=1, help='0: Disable log information saver, 1: Activate log information saver')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_model', type=int, default=1, help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--topks', nargs='?', default='[10, 20, 50, 100, 200]', help='Value of top-K')
    parser.add_argument('--early_stop', type=int, default=50, help='Early stop patience.')
    parser.add_argument('--dataset', type=str, default='taobao', help='Choose a dataset from {ijcai, taobao}')
    parser.add_argument('--data_path', type=str, default='data/', help='Input data path.')
    parser.add_argument('--cache_flag', type=int, default=1,help='0: Disable cache train and test data, 1: Activate cache train and test data')
    parser.add_argument('--multi', type=int, default=1, help='Multiplier for the result')
    parser.add_argument('--n_behavior', type=int, default=4, help='Number of behaviors')
    parser.add_argument('--n_train_neg', default=4, type=int, help='Number of item for each train data sampling.')
    parser.add_argument('--n_train_sample', default=40, type=int, help='Number of samples for each user while training.')
    parser.add_argument('--n_train_user', default=120000, type=int, help='Number of users for training.')
    parser.add_argument('--n_test_epoch', default=1, type=int, help='Number of epoch to test while training.')
    parser.add_argument('--batch_size_train', type=int, default=2048, help='Batch size in training.')
    parser.add_argument('--batch_size_test', type=int, default=512, help='Batch size in testing.')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument("--gamma1", type=float, default=1e-5, help="Synergy alignment loss regularization rate")
    parser.add_argument("--gamma2", type=float, default=1e-5, help="BPR loss regularization rate")
    parser.add_argument('--n_layer', default=3, type=int, help='Number of learning layers.')
    parser.add_argument('--embedding_size', type=int, default=32, help='Embedding size.')
    parser.add_argument('--self_loop_weight', type=float, default=1, help='Self loop weight')
    parser.add_argument('--lamda', type=float, default=0.9, help='Weight of user-based scoring, [0., 1.).')
    parser.add_argument('--msg_dropout', type=float, default=0.2, help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 0: no dropout.')
    parser.add_argument('--is_align_target', type=bool, default=False, help='Whether or not to apply synergy alignment loss to the target behavior')

    return parser.parse_args()


args = parse_args()