import os
import os.path as osp

from tqdm import tqdm

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loss_recorder import LossRecorder

from paired_dataset import create_dataset_from_args
from modules import RunningNormalizeModel

from utils.running_statics import RunningStatistics
from unified_utils import model_dispatch

import data_process.data_processors as data_processors


def main():
    model, remaining_args, model_str = model_dispatch()
    option_parser = model.TrainOption()
    args = option_parser.parse_args(remaining_args)
    if args.data_processor_type != 'Unknown':
        data_processor = data_processors.processor_dispatcher(args.data_processor_type)
    else:
        data_processor = model.ExpectedDataProcessor()

    Save = args.save
    utility.MakeDirectory(Save)
    with open(osp.join(Save, "args.txt"), "w") as file:
        file.write(model_str + " " + option_parser.text_serialize(args))
    args = option_parser.post_process(args)

    log_dir = osp.join(Save, 'log')
    if os.path.exists(log_dir) and 'test' not in log_dir:
        print('log dir exists, remove it [y/n]?')
        if input() != 'y':
            print('exit')
            return
    if osp.exists(log_dir):
        os.system(f'rm -rf {log_dir}')
    summary_writer = SummaryWriter(log_dir)
    loss_recorder = LossRecorder(summary_writer)

    motion_data = create_dataset_from_args(args)

    # Build network model
    network = model.create_model_from_args(args, motion_data, data_processor)

    if args.running_normalize:
        network = RunningNormalizeModel(network)
    network = utility.ToDevice(network)

    params = network.parameters()

    # Setup optimizer and loss function
    optimizer = adamw.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=args.batch_size,
                                                      epoch_size=len(motion_data),
                                                      restart_period=args.restart_period, t_mult=args.restart_mult,
                                                      policy="cosine", verbose=True)
    data_loader = DataLoader(motion_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                               pin_memory=True)

    r_s = RunningStatistics()

    for epoch in range(args.epochs):
        scheduler.step()
        n_iter_total = len(data_loader)
        if args.debug:
            n_iter_total = 3

        if args.use_tqdm:
            loop = tqdm(range(n_iter_total))
        else:
            loop = range(n_iter_total)
        it_dataloader = iter(data_loader)
        for n_iters in loop:
            train_batch = next(it_dataloader)
            network.train()

            input_data, follow_output = data_processor.ready_input_output(train_batch, args, motion_data)

            if args.noise_level > 0:
                input_data = input_data + torch.randn_like(input_data) * args.noise_level

            input_data = utility.ToDevice(input_data)
            follow_output = utility.ToDevice(follow_output)

            r_s.update(input_data)

            losses, _ = network.learn(input_data, follow_output)

            loss_total = sum([losses[k] * getattr(args, f'lambda_{k}') for k in losses], torch.tensor(0.0).to(input_data.device))

            for k in losses:
                loss_recorder.add_scalar(f'loss_{k}', losses[k].item())
            loss_recorder.add_scalar(f'loss_total', loss_total.item())

            loss_descript = ' '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
            loss_descript = f'total: {loss_total.item():.4f} ' + loss_descript
            if args.use_tqdm:
                loop.set_description(loss_descript)

            if len(losses) > 0:   # Only do it if the running normalization has been initialized
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            if not args.use_tqdm and n_iters % 500 == 0:
                print(f'Epoch {epoch + 1}/{args.epochs} Iter {n_iters}/{n_iter_total} {loss_descript}')

        scheduler.batch_step()
        if args.debug:
            break

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save(network.state_dict(), f'{Save}/{epoch + 1:04d}.pt')

        loss_recorder.epoch()


if __name__ == '__main__':
    main()
