# encoding:utf-8
import os
import sys
import time
import numpy as np
import random

import torch
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

from utils import helper
from utils import evaluate
from utils import data_utils
from utils.TimeLogger import init_log, log
from utils.parser import args
from utils.constvars import CONST

from model.BPRMF import BPRMF
from model.SWGCN import SWGCN

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    device_name = 'cuda:' + str(args.gpu_id) if args.gpu_id >= 0 else 'cpu'
    args.device = torch.device(device_name)

    topks = list(eval(args.topks))

    start_train_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    model_path = f'{args.model_path}/{args.model}/{args.exp_name}/{args.dataset}-seed-{seed}'
    if args.load_model == 0:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    init_log(start_train_time, model_path)

    log('args: {}'.format(args), save=args.save_log)

    ############################## PREPARE DATASET ##########################
    log('start load all data', save=args.save_log)
    train_mats, test_data, validation_data, n_user, n_item = data_utils.load_all(args.dataset)  # 因为后面将validation
    log('n_user: {}, n_item: {}'.format(n_user, n_item), save=True)
    log('end load all data', save=args.save_log)

    # construct the train and test datasets
    train_mask = train_mats[-1]  # Training data only target behavior interactions
    train_dataset = data_utils.TrainData(n_user, n_item, args.n_train_neg, train_mats, None, args.n_train_user, args.n_train_sample, is_training=True)
    test_dataset = data_utils.TestDataset(test_data, train_mask)
    validation_dataset = data_utils.TestDataset(validation_data, train_mask)

    test_n_worker = args.n_worker
    validation_loader = data.DataLoader(validation_dataset, batch_size=args.batch_size_test, pin_memory=True, shuffle=False, num_workers=test_n_worker)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size_test, pin_memory=True, shuffle=False, num_workers=test_n_worker)

    ########################### CREATE MODEL #################################
    log('init {} model'.format(args.model), save=True)
    if args.model == CONST.BPRMF:
        model = BPRMF(args, n_user, n_item)
    elif args.model == CONST.SWGCN:
        model = SWGCN(args, train_mats, n_user, n_item)
    else:
        log('Model is not exist.', save=True)
        sys.exit(1)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(f'{model_path}/tensorboard/{start_train_time}')  # for visualization

    ########################### TRAINING #####################################
    start_epoch, best_epoch, best_hr, best_ndcg = 0, 0, 0, 0
    best_loss = 1e8
    validation_metrics = None
    loss = None

    # early stop
    stopping_step = 0

    if args.load_model:
        model_save_path = '{}/last.pt'.format(model_path)
        if not os.path.exists(model_save_path):
            assert not os.path.exists(model_save_path), 'model is not exist'
            return
        checkpoint = torch.load(model_save_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        validation_metrics = checkpoint['metrics']  # {HR: #, NDCG: #}
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_hr = validation_metrics[topks[0]]['HR']
        best_ndcg = validation_metrics[topks[0]]['NDCG']

        start_epoch = best_epoch + 1
        log('trained epoch {:03d}: HR@{:02d}: {:.5f}, NDCG@{:02d}: {:.5f}, best_loss: {}'.format(best_epoch, topks[0],best_hr, topks[0], best_ndcg, best_loss),
            save=args.save_log)
        log('start epoch: {:03d}'.format(start_epoch), save=True)

    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        start_time = time.time()

        log('start generate train data', save=True)
        train_dataset.generate_train_data(args.n_behavior - 1, epoch, args.dataset, args.cache_flag)  # Take only the target behavior as training data
        log('end generate train data', save=True)


        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size_train, pin_memory=True, shuffle=True, num_workers=args.n_worker)
        batch = 0
        for users, pos_items, neg_items in train_loader:
            if batch % 100 ==0:
                log('start %d train batch' % batch, save=args.save_log)
            users = users.to(args.device)
            pos_items = pos_items.to(args.device)
            neg_items = neg_items.to(args.device)

            if args.model == CONST.BPRMF:
                user_embeddings, pos_item_embeddings, neg_item_embeddings = model(users, pos_items, neg_items)
                loss, fun_loss, _ = model.create_bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)

            elif args.model == CONST.SWGCN:
                user_embeddings, pos_item_embeddings, neg_item_embeddings, similarity_list, total_score_list = model(users, pos_items, neg_items)
                loss = model.create_joint_loss(user_embeddings,
                                               pos_item_embeddings,
                                               neg_item_embeddings,
                                               similarity_list, total_score_list)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('data/batch_loss', loss.item(), epoch * (len(train_dataset) // args.batch_size_train) + batch)

            batch += 1

        log('train loss: {}'.format(loss), save=True)
        writer.add_scalar('data/epoch_loss', loss.item(), epoch)

        ### validation ###
        if epoch % args.n_test_epoch == 0 or epoch == args.n_epoch - 1:
            model.eval()
            with torch.no_grad():
                log('start evaluate', save=True)
                validation_metrics = evaluate.metric_all(args.model, model, validation_loader, n_item, topks,
                                                         args.device, args.multi)
                log('end evaluate', save=True)
                log('validation metircs: {}'.format(validation_metrics), save=True)

                HR = validation_metrics[topks[0]]['HR']
                NDCG = validation_metrics[topks[0]]['NDCG']

                elapsed_time = time.time() - start_time
                log("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)), save=True)


                stopping_step, should_stop = helper.early_stopping(HR, best_hr, stopping_step,expected_order='acc', flag_step=np.floor(args.early_stop // args.n_test_epoch))

                if should_stop:
                    log("Early stopping is trigger at step: {}.".format(np.floor(args.early_stop // args.n_test_epoch)),
                        save=True)
                    break

            helper.ensureDir(model_path)
            if best_hr < HR:
                best_loss = loss
                best_epoch = epoch
                best_hr =  HR
                best_ndcg = NDCG
                torch.save({
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': validation_metrics
                }, '{}/best.pt'.format(model_path))

            torch.save({
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': validation_metrics
            }, '{}/last.pt'.format(model_path))


    writer.close()  # to do 如果用writer记录超参数结果，那么在超参数调整期间不能在这里close

    log("End. Best epoch {:03d}: HR@{:03d} = {:.5f}, NDCG@{:03d} = {:.5f}".format(best_epoch, topks[0], best_hr, topks[0], best_ndcg), save=True)

    ###  test best.pt  ###
    model.eval()
    with torch.no_grad():
        model_save_path = '{}/best.pt'.format(model_path)
        if not os.path.exists(model_save_path):
            assert not os.path.exists(model_save_path), 'model is not exist'
            return
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        log('start test best model', save=True)
        best_test_metrics = evaluate.metric_all(args.model, model, test_loader, n_item, topks, args.device, args.multi)
        log('end test best model', save=True)
        log('test best model metircs: {}'.format(best_test_metrics), save=True)

    return start_train_time, best_test_metrics


if __name__ == '__main__':
    start_train_time, best_test_metrics = main()
