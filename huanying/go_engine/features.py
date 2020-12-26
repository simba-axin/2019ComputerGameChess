"""
AlphaGo所使用的特征，按重要性排序。 总共构成神经网络输入的8个平面共48个特征
特征（Feature）                     平面数量（num_plans）   说明（Notes）
颜色（Stone colour）                3                      我方，对方颜色，以及空点（Player stones; oppo. stones; empty）
1（Ones）                           1                      全1平面（Constant plane of 1s）辨别边界
轮次（Turns since last move）       8                      每个落子过后经过的轮次（How many turns since a move played）
气（Liberties）                     8                      每个落子气的数量（Number of liberties）
打吃（Capture size）                8                      对手被打吃的数目（How many opponent stones would be captured）
被打吃（Self-atari size）           8                      自己被打吃的数目（How many own stones would be captured）
落子后的气（Liberties after move）  8                      每个落子刚落之后的气（Number of liberties after this move played）
征子有利（ladder capture）          1                      落子是否征子有利（Whether a move is a successful ladder cap）
征子逃脱（Ladder escape）           1                      落子是否征子逃脱（Whether a move is a successful ladder escape）
合法性（Sensibleness）              1                      落子是否合法并且没有填自己的眼（Whether a move is legal + doesn't fill own eye）
0（Zeros）                          1                      全零平面（Constant plane of 0s）


(Because of convolution w/ zero-padding, this is the only way the NN can know where the edge of the board is!!!)
All features with 8 planes are 1-hot encoded, with plane i marked with 1
only if the feature was equal to i. Any features >= 8 would be marked as 8.
限制一个平面的特征数量，最多为8
"""

import numpy as np
import go_engine.go as go
from utils.go_utils import product

# Resolution/truncation limit for one-hot features
P = 8


def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.uint8)  #unit8:无符号八位整形
    capped = np.minimum(feature, planes)
    onehot_index_offsets = np.arange(0, product(onehot_features.shape), planes) + capped.ravel()
    # A 0 is encoded as [0,0,0,0], not [1,0,0,0], so we'll
    # filter out any offsets that are a multiple of $planes
    # A 1 is encoded as [1,0,0,0], not [0,1,0,0], so subtract 1 from offsets
    nonzero_elements = (capped != 0).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features


def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

#返回三维的棋盘上棋子颜色的特征
@planes(3)
def stone_color_feature(position):
    board = position.board
    features = np.zeros([go.N, go.N, 3], dtype=np.uint8)
    if position.to_play == go.BLACK:
        features[board == go.BLACK, 0] = 1
        # board里的全部位置依次和go.black进行比较,返回当前位置是否相等
        features[board == go.WHITE, 1] = 1
    else:
        features[board == go.WHITE, 0] = 1
        features[board == go.BLACK, 1] = 1

    features[board == go.EMPTY, 2] = 1
    return features

#全1平面,辨别边界
@planes(1)
def ones_feature(position):
    return np.ones([go.N, go.N, 1], dtype=np.uint8)

#最近的行棋特征
@planes(P)
def recent_move_feature(position):
    onehot_features = np.zeros([go.N, go.N, P], dtype=np.uint8)
    for i, player_move in enumerate(reversed(position.recent[-P:])):
        _, move = player_move # unpack the info from position.recent
        if move is not None:
            onehot_features[move[0], move[1], i] = 1
    return onehot_features

#每个落子气的数量
@planes(P)
def liberty_feature(position):
    return make_onehot(position.get_liberties(), P)

# 将要提子的特征
@planes(P)
def would_capture_feature(position):
    features = np.zeros([go.N, go.N], dtype=np.uint8)
    for g in position.lib_tracker.groups.values():
        if g.color == position.to_play:
            continue
        if len(g.liberties) == 1:
            last_lib = list(g.liberties)[0]
            # += because the same spot may capture more than 1 group.
            features[last_lib] += len(g.stones)
    return make_onehot(features, P)

DEFAULT_FEATURES = [
    stone_color_feature,
    ones_feature,
    liberty_feature,
    recent_move_feature,
    would_capture_feature,
]


def extract_features(position, features=DEFAULT_FEATURES):
    return np.concatenate([feature(position) for feature in features], axis=2)


def bulk_extract_features(positions, features=DEFAULT_FEATURES):
    num_positions = len(positions)
    num_planes = sum(f.planes for f in features)
    output = np.zeros([num_positions, go.N, go.N, num_planes], dtype=np.uint8)
    for i, pos in enumerate(positions):
        output[i] = extract_features(pos, features=features)
    return output
