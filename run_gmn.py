import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from model_heterogmn import MLP_simple
from model_heterogmn import GMN_ORTHO, APPmem, APPmem2
from load_data_gmn import load_dataset
from parse_gmn import *
from utils import eval_acc



def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    out, proto_loss, entro_loss, regu_loss = model(data.x, data.edge_index)

    class_loss = F.nll_loss(out[train_mask], data.y[train_mask])
    total_loss = class_loss + args.alpha * proto_loss + args.beta * entro_loss + args.regu * regu_loss
    total_loss.backward()
    return total_loss


@torch.no_grad()
def test(model):
    model.eval()
    out, proto_loss, entro_loss, regu_loss = model(data.x, data.edge_index)


    val_pred = out[val_mask].max(1)[1]
    val_acc = val_pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()
    val_class = F.nll_loss(out[val_mask], data.y[val_mask])
    val_loss = val_class + args.alpha * proto_loss + args.beta * entro_loss + args.regu * regu_loss

    test_pred = out[test_mask].max(1)[1]
    test_acc = test_pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()

    return val_acc, test_acc, val_loss, val_class, proto_loss, entro_loss



def train_mlp(model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = mlp(data.x)
    out = F.log_softmax(out, dim=1)

    if args.dataname in ['twitch', 'fb100']:
        true_label = F.one_hot(data.y[train_mask], data.y.max() + 1).to(torch.float)
    else:
        true_label = data.y[train_mask]
    loss_mlp = criterion(out[train_mask], true_label)
    loss_mlp.backward()
    optimizer.step()


@torch.no_grad()
def test_mlp(model):
    model.eval()
    with torch.no_grad():
        accs = []
        out = mlp(data.x)
        out = F.log_softmax(out, dim=1)
        for mask in (val_mask, test_mask):
            # pred = logits[mask].max(1)[1]
            acc = eval_func(data.y[mask].unsqueeze(1), out[mask])
            accs.append(acc)
        return accs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load Data
    data = load_dataset(args.dataname, args.train_prop, args.valid_prop, args.num_masks)
    data = data.to(device)
    print('-' * 150)
    print(str(args))
    print(data)




    criterion = torch.nn.NLLLoss()
    eval_func = eval_acc

    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1
    num_features = data.x.shape[1]


    cluster_labels = 0
    centers = 0


    best_val_acc_multi, test_acc_multi = [], []
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]


        print('Pre-train MLP:')
        mlp = MLP_simple(num_features, args.mlp_hidden, num_classes).to(device)
        optimizer2 = torch.optim.Adam(mlp.parameters(), lr=args.pre_lr, weight_decay=5e-4)
        best_val_acc = test_acc = 0

        for epoch in range(args.pretrain_epochs + 1):
            train_mlp(mlp, optimizer2)

            if epoch % 50 == 0:
                val_acc, tmp_test_acc = test_mlp(mlp)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, best_val_acc, test_acc))


        # Get y_pseudo label
        y_pseudo = mlp(data.x).max(1)[1]
        y_pseudo[train_mask] = data.y[train_mask]


        print('Train: Memory Network')
        gmn = GMN_ORTHO(data.x, data.edge_index, num_nodes, num_features, num_classes, y_pseudo, centers, cluster_labels,
                        args.K, args.hidden, args.dropout, args.ppr_alpha, args.local_stat_num, args.memory_hidden, device).to(device)
        optimizer3 = torch.optim.Adam([
                                      {'params': gmn.parameters(), 'weight_decay': args.weight_decay},
                                      {'params': gmn.memory}
                                      ], lr=args.lr, weight_decay=0.000)


        best_val_acc = test_acc = 0
        best_val_loss = float('inf')
        best_tuple = (0, 0, 0)

        for epoch in range(args.epochs + 1):
            train_loss = train(gmn, optimizer3)

            val_acc, tmp_test_acc, val_loss, val_class, memory_loss, entro_loss = test(gmn)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
                best_tuple = (val_class, memory_loss, entro_loss)

            if epoch % 50 == 0:
            # val_acc, tmp_test_acc = test(gmn)
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     test_acc = tmp_test_acc
                best_class_loss, best_memory_loss, best_ortho_loss = best_tuple
                log = 'Train_loss:{:.4f}, Epoch:{:03d}, best_val_acc:{:.4f}, test_acc:{:.4f}, best_val_loss:{:.4f} = (best_class_loss:{:.4f}, memory_loss:{:.4f}, entro_loss:{:.4f})'
                print(log.format(train_loss, epoch, best_val_acc, test_acc, best_val_loss, best_class_loss, best_memory_loss, best_ortho_loss))


        best_val_acc_multi.append(best_val_acc)
        test_acc_multi.append(test_acc)

    # Process results
    best_val_acc_multi.append(np.mean(best_val_acc_multi))
    test_acc_multi.append(np.mean(test_acc_multi))
    best_val_acc_multi = (np.array(best_val_acc_multi) * 100).reshape(-1,1)
    test_acc_multi = (np.array(test_acc_multi) * 100).reshape(-1, 1)

    # Save results
    save_args = '_'.join(str(i) for i in [args.seed, args.hidden, args.dropout, args.lr, args.weight_decay,
                                          args.num_layers, args.K, args.alpha, args.beta, args.regu])

    result = np.around(np.concatenate((best_val_acc_multi, test_acc_multi), 1), decimals=3)
    print(result)
    print(np.std(result[:,1]))
    path = os.path.join('result', args.dataname, 'gmn_ortho', save_args)
    file_name = path + '.csv'
    print(file_name)
    np.savetxt(file_name, result, fmt='%.03f', delimiter=',')



