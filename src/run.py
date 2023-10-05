# import haienv
# haienv.set_env('qmsum')
import argparse
from torch.utils.data import DataLoader
from data_builder import OurDataset
from models.model import BaseModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for data_path
    parser.add_argument('-data_path', default='', type=str)
    parser.add_argument('-val_save_file', default='', type=str)
    parser.add_argument('-test_save_file', default='', type=str)

    # for model settings
    parser.add_argument('-model', default='fidbart', type=str)
    parser.add_argument('-checkpoint', default='', type=str)
    parser.add_argument('-segment_interaction', default='no', type=str)
    parser.add_argument('-knowledge_aware', default='', type=str)

    # for training
    parser.add_argument('-log_name', default='BART_large', type=str)
    parser.add_argument('-gpus', default='0', type=str)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-learning_rate', default=3e-5, type=float)
    parser.add_argument('-adafactor', action='store_true')
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-warmup', type=int, default=20)
    parser.add_argument('-label_smoothing', type=float, default=0.1)
    parser.add_argument('-grad_accum', type=int, default=10)
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-do_train', action='store_true')
    parser.add_argument('-do_test', action='store_true')
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', type=float, default=1)

    # data settings
    parser.add_argument('-max_input_len', type=int, default=512)
    parser.add_argument('-number_of_segment', type=int, default=16)
    parser.add_argument('-max_output_len', type=int, default=64)
    parser.add_argument('-min_output_len', type=int, default=64)
    parser.add_argument('-n_beams', type=int, default=5)
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('-length_penalty', type=float, default=0.5)

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(args.log_name)
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('terminal_output/' + args.log_name.split('/')[-1] + str(args.random_seed))
    print(args)
    # save checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='validation_Rouge/rouge1_F1',
                                          save_last=False,
                                          save_top_k=1,
                                          mode='max')

    # make trainer
    if args.checkpoint == 'None':
        resume_checkpoint = None
    else:
        resume_checkpoint = args.checkpoint
    trainer = Trainer(
                    #   deterministic=True,
                      num_sanity_val_steps=4,
                      resume_from_checkpoint=resume_checkpoint,
                      logger=logger,
                      devices='-1',
                      accelerator='gpu',
                      strategy="deepspeed_stage_2",
                      precision=16,
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accum,
                      log_every_n_steps=1,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback])


    # make dataloader & model
    train_set = OurDataset(args, 'train')
    val_set = OurDataset(args, 'val')
    test_set = OurDataset(args, 'test')
    train_loader = DataLoader(dataset=train_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=4, \
                                    shuffle=True, \
                                    collate_fn=train_set.collate_fn)
    val_loader = DataLoader(dataset=val_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=4, \
                                    shuffle=False, \
                                    collate_fn=val_set.collate_fn)
    test_loader = DataLoader(dataset=test_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=4, \
                                    shuffle=False, \
                                    collate_fn=test_set.collate_fn)
    model = BaseModel(args)

    # Fit the instantiated model to the data
    if args.do_train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.do_test:
        # model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, ckpt_path=args.checkpoint, dataloaders=test_loader)

