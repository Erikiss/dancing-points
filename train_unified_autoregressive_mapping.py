from pathlib import Path

from tqdm import tqdm

import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loss_recorder import LossRecorder

from paired_dataset import create_dataset_from_args
from modules import RunningNormalizeModel
from runtime_utils import prepare_log_dir, resolve_device, seed_everything, seed_worker
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

    serialized_args = model_str + " " + option_parser.text_serialize(args)
    args = option_parser.post_process(args)
    seed_everything(args.seed, deterministic=bool(args.deterministic))
    device = resolve_device(args.device)
    print(f'Device: {device}; seed: {args.seed}; deterministic: {bool(args.deterministic)}')

    motion_data = create_dataset_from_args(args)

    Save = Path(args.save)
    log_dir = prepare_log_dir(Save / 'log', overwrite=bool(args.overwrite_log))
    with open(Save / "args.txt", "w") as file:
        file.write(serialized_args)

    summary_writer = SummaryWriter(str(log_dir))
    loss_recorder = LossRecorder(summary_writer)

    # Build network model
    network = model.create_model_from_args(args, motion_data, data_processor)

    if args.running_normalize:
        network = RunningNormalizeModel(network)
    network = network.to(device)

    params = network.parameters()

    # Setup optimizer and loss function
    optimizer = adamw.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=args.batch_size,
                                                      epoch_size=len(motion_data),
                                                      restart_period=args.restart_period, t_mult=args.restart_mult,
                                                      policy="cosine", verbose=True)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    use_pinned_memory = device.type == 'cuda'
    data_loader = DataLoader(
        motion_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=use_pinned_memory,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=data_generator,
    )

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

            input_data = input_data.to(device, non_blocking=use_pinned_memory)
            follow_output = follow_output.to(device, non_blocking=use_pinned_memory)

            losses, _ = network.learn(input_data, follow_output)

            loss_total = sum(
                [losses[k] * getattr(args, f'lambda_{k}') for k in losses],
                torch.zeros((), device=input_data.device),
            )

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
            torch.save(network.state_dict(), Save / f'{epoch + 1:04d}.pt')

        loss_recorder.epoch()


if __name__ == '__main__':
    main()
