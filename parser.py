
def parser_add_main_args(parser):

    # Data
    parser.add_argument('--dataname', type=str, default='squirrel')
    parser.add_argument('--num_masks', type=int, default=10, help='number of masks')
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')

    # Training
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--pre_lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=330)

    # Model
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--times', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=500)
    parser.add_argument('--memory_hidden', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--regu', type=float, default=100)
    parser.add_argument('--mlp_hidden', type=int, default=512)
    parser.add_argument('--local_stat_num', type=int, default=4)
    parser.add_argument('--ppr_alpha', type=float, default=0.25)


