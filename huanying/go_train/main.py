import argparse
import argh
from contextlib import contextmanager
import os
import random
import re
import sys
import time
sys.path.append("..")
import go_train.gtp as gtp_lib

from go_engine.netEngine import PolicyNetWork
from go_engine.aiEngine import PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS
from utils.load_data_sets import DataSet, parse_data_sets

#正则表达式，匹配文件
TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

#两个计时器
@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))


def gtp(strategy, read_file=None):
    n = PolicyNetWork(use_cpu=False)
    if strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
    else:
        sys.stderr.write("Unknown strategy")
        sys.exit()
    gtp_engine = gtp_lib.Engine(instance)
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()

# def run():  想用sabaki处理才写的函数，没能成功
#     read_file = "F:\\PhantomGo000\\go_data\\model\\savemodel"
#     n = PolicyNetWork(use_cpu=True)
#     instance = PolicyNetworkBestMovePlayer(n, read_file)
#     gtp_engine = gtp_lib.Engine(instance)
#     sys.stderr.write("GTP engine ready\n")
#     sys.stderr.flush()
#     while not gtp_engine.disconnect:
#         inpt = input()
#         # handle either single lines at a time
#         # or multiple commands separated by '\n'
#         try:
#             cmd_list = inpt.split("\n")
#         except:
#             cmd_list = [inpt]
#         for cmd in cmd_list:
#             engine_reply = gtp_engine.send(cmd)
#             sys.stdout.write(engine_reply)
#             sys.stdout.flush()

def preprocess(*data_sets, processed_dir="..\go_data\pre_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    test_chunk, training_chunks = parse_data_sets(*data_sets)
    print("%s的数据作为test(测试集),剩下的数据作为训练集" % len(test_chunk))  # , file=sys.stderr)

    print("制作test chunk(测试集)")
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)

    print("制作train chunk(训练集)")
    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("已经制作了%s训练集" % (i + 1))
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    # print("%s chunks written" % (i+1))

def train(processed_dir, read_file=None, save_file=None, epochs=2, logdir=None, checkpoint_freq=1000):
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(processed_dir, fname)
        for fname in os.listdir(processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    save_file = os.path.join(os.getcwd(), save_file)
    n = PolicyNetWork()
    try:
        n.initialize_variables(save_file)
    except:
        n.initialize_variables(None)
    if logdir is not None:
        n.initialize_logging(logdir)
    last_save_checkpoint = 0
    for i in range(epochs):
        random.shuffle(train_chunk_files)
        for file in train_chunk_files:
            print("Using %s" % file)
            train_dataset = DataSet.read(file)
            train_dataset.shuffle()
            with timer("training"):
                n.train(train_dataset)
            n.save_variables(save_file)
            if n.get_global_step() > last_save_checkpoint + checkpoint_freq:
                with timer("test set evaluation"):
                    n.check_accuracy(test_dataset)
                last_save_checkpoint = n.get_global_step()



parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
train(r'D:\计算机博弈大赛\huanying\go_data\pre_data',
      save_file=r'D:\计算机博弈大赛\huanying\go_data\model\savedmodel')
