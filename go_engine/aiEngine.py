import copy
import math
import random
import time
import sys
sys.path.append("..")
import go_data.globalData as data
import numpy as np
import go_train.gtp as gtp
import go_engine.go as go
import utils.go_utils as utils

# 对于可能的行棋策略进行排序，返回排好序列的坐标值
def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    return sorted(coords, key=lambda c: probability_array[c], reverse=True)

# 与gtp标准协议进行转换，方便判断输出和输入
def translate_gtp_colors(gtp_color):
    if gtp_color == data.BLACK:
        return go.BLACK
    elif gtp_color == data.WHITE:
        return go.WHITE
    else:
        return go.EMPTY

# 通过行棋步骤判断行棋是否合法（是否为对方的眼，或者是否符合规定）返回值为一个move
def is_move_reasonable(position, move):
    return position.is_move_legal(move) and go.is_eyeish(position.board, move) != position.to_play

# 通过对这一步棋的估值网络的结果的转换，找到最优的下一步预测。返回值为一个move
def select_most_likely(position, move_probabilities):
    for move in sorted_moves(move_probabilities):
        if is_move_reasonable(position, move):
            return move
    return None

# 随机权重用来检测可能行棋中的所有棋，返回一个随机的行棋，如果选择的行棋不合法，找到下一个最优预测进行返回。（move）
def select_weighted_random(position, move_probabilities):
    selection = random.random()
    selected_move = None
    current_probability = 0
    # move为一个坐标，为move_probabilities的取值位置。move_prob为每一个取值。
    for move, move_prob in np.ndenumerate(move_probabilities):
        current_probability += move_prob
        if current_probability > selection:
            selected_move = move
            break
    if is_move_reasonable(position, selected_move):
        return selected_move
    else:
        # 如果后面的行棋违法则撤回
        print("Using fallback move; position was %s\n, selected %s" % (
            position, selected_move))
        return select_most_likely(position, move_probabilities)

# gtp接口，用来进行输出和输入的操作规范。
class GtpInterface(object):
    def __init__(self):
        self.size = 9
        self.position = None   #坐标
        self.pre_position = None  # 慢一步的position，用来处理幻影围棋的提子信息
        self.komi = 6.5
        self.clear()

    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    # 更新position信息
    def clear(self):
        self.position = go.Position(komi=self.komi)

    # 判断一轮棋，更新弃子的情况
    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    # 对于当前一步进行行棋更新，返回行棋完成后的新的棋盘信息。
    def make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)  # 先变化这个点的坐标为标准协议
        self.accomodate_out_of_turn(color)
        self.position = self.position.play_move(coords, color=translate_gtp_colors(color))
        return self.position is not None

    # 对于对手的一步棋进行跟新，跟新效果和上一个函数一样，不过加入一手对手弃子之后的更新方法。
    def pre_make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)
        if not translate_gtp_colors(color) == self.pre_position.to_play:
            self.pre_position.flip_playerturn(mutate=True)
        self.pre_position = self.pre_position.play_move(coords, color=translate_gtp_colors(color))
        return self.pre_position is not None

    # 获取当前行棋
    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    def suggest_move(self, position):
        raise NotImplementedError

# 通过模型判断，寻找最优下棋策略
class PolicyNetworkBestMovePlayer(GtpInterface):
    def __init__(self, policy_network, read_file):
        self.policy_network = policy_network
        self.read_file = read_file
        super().__init__()

    # 更新
    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # 更新价值网络，确保策略函数用的是最新的估值网络。
        # 这样在下棋的过程中，也能训练网络，并且可以确保使用的是最优的网络。
        self.policy_network.initialize_variables(self.read_file)

    # 通过对于这一步的在估值网络中的搜索，进行更新返回最优的行棋策略
    def suggest_move(self, position):
        if position.recent and position.n > 100 and position.recent[-1].move == None:
            # 对手一直pass的时候，自己也pass。
            return None
        move_probabilities = self.policy_network.run(position)
        return select_most_likely(position, move_probabilities)

# 快速走子策略，随机走子用来结合神经网络的判断
class RandomPlayer(GtpInterface):
    def suggest_move(self, position):
        possible_moves = go.ALL_COORDS[:]
        random.shuffle(possible_moves)
        for move in possible_moves:
            if is_move_reasonable(position, move):
                return move
        return None

class PolicyNetworkRandomMovePlayer(GtpInterface):
    def __init__(self, policy_network, read_file):
        self.policy_network = policy_network
        self.read_file = read_file
        super().__init__()

    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # Ensure that the player is using the latest version of the network
        # so that the network can be continually trained even as it's playing.
        self.policy_network.initialize_variables(self.read_file)

    def suggest_move(self, position):
        if position.recent and position.n > 100 and position.recent[-1].move == None:
            # 如果对手一直pass处理方法
            return None
        move_probabilities = self.policy_network.run(position)
        return select_weighted_random(position, move_probabilities)

# 搜索常量
c_PUCT = 5


class MCTSNode():
    '''
    MCTSNode有两种状态节点:plain和expand。
    一个普通的MCTSNode只知道它的Q + U值，这就使得plain节点可以随意扩展
    扩展后，MCTSNode还必须存储该节点的实际位置，
    以及通过策略网络跟踪移动/概率。
    每个后续步骤都被实例化为一个普通的MCTSNode。
    Q：当前节点的汇总平均值
    U：当前节点的奖励估值
    '''
    @staticmethod
    # 根结点的创建以及扩展
    def root_node(position, move_probabilities):
        node = MCTSNode(None, None, 0)
        node.position = position
        node.expand(move_probabilities)
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # 父节点位置
        self.move = move # 找到这个节点的移动操作
        self.prior = prior# 优先
        self.position = None # 对于扩展的position存储以及计算
        self.children = {} # MCTS节点的子节点的集合，shape为map
        self.Q = self.parent.Q if self.parent is not None else 0 # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus优先值
        self.N = 0 # number of times node was visited探索的层数

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    # 计算当前点的权重
    def action_score(self):
        # 在加入价值网络之前必须要计算q的值
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        return self.Q + self.U

    # 判断是否为扩张节点
    def is_expanded(self):
        return self.position is not None

    # 判断估算的点是否满足条件
    def compute_position(self):
        self.position = self.parent.position.play_move(self.move)
        return self.position

    # 把所有的可能性节点变成当前节点的子节点
    def expand(self, move_probabilities):
        self.children = {move: MCTSNode(self, move, prob)
            for move, prob in np.ndenumerate(move_probabilities)}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSNode(self, None, 0)

    # 回述权值，从子节点一直回述到根结点，返回权值为计算权值的相反数，因为估值网络的判断要求和计算出来的相反
    def backup_value(self, value):
        self.N += 1
        if self.parent is None:
            # 当没有点用于更新u，q的时候，到达根结点，并返回。
            return
        self.Q, self.U = (
            self.Q + (value - self.Q) / self.N,
            c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N,
        )
        # must invert, because alternate layers have opposite desires
        self.parent.backup_value(-value)

    # 通过最大值来选择分支节点
    def select_leaf(self):
        current = self
        while current.is_expanded():
            current = max(current.children.values(), key=lambda node: node.action_score)
        return current


class MCTS(GtpInterface):
    def __init__(self, policy_network, read_file, seconds_per_move=5):
        self.policy_network = policy_network      #估值网络
        self.seconds_per_move = seconds_per_move  #每一步策略的得分
        self.max_rollout_depth = go.N * go.N * 3  #最大输出深度
        self.read_file = read_file                #读取估值网络模型
        super().__init__()

    #更新网络
    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # 确保对局中使用的是最新版本的网络
        # 这样，即使在游戏中，网络也可以不断地进行训练。
        self.policy_network.initialize_variables(self.read_file)

    def suggest_move(self, position):
        if position.caps[0] + 50 < position.caps[1]:
            return gtp.RESIGN
        start = time.time()
        # 获取当前这一步的特征概率，进行判断搜索
        move_probs = self.policy_network.run(position)
        # 创建当前这一步的根结点
        for i in range(9):
            for j in range(9):
                x = go.is_eyeish(position.board, (i, j))
                #print(x)
                if position.board[i][j] != go.EMPTY:
                    move_probs[i][j] = float(0)
                elif x != None and x != position.to_play:
                    move_probs[i][j] = float(0)
        # print(move_probs)
        root = MCTSNode.root_node(position, move_probs)
        # print('进入蒙特卡洛搜索树')
        while time.time() - start < self.seconds_per_move:
            self.tree_search(root)

        # print('蒙特卡洛搜索树结束')

        # 如果自己拒绝了pass这一步，这个ai会开始填充自己的眼，所以要进行判断，如果是非法的行棋要找出来。
        # 返回值为一个根节点的所有子节点中的最大的那一个
        if position.n < 45:
            # print(str(root.children[(4, 4)].Q))
            # print(max(root.children.keys(), key=lambda move, root=root: root.children[move].Q))
            return max(root.children.keys(), key=lambda move, root=root: root.children[move].N)
        else:
            while True:
                max_move = max(root.children.keys(), key=lambda move, root=root: root.children[move].N)
                # print(move_probs)
                # root.children.pop(max_move)
                x = max_move[0]
                y = max_move[1]
                if go.is_eyeish(position.board, max_move) != position.to_play and move_probs[x][y] != float(0):
                    return max(root.children.keys(), key=lambda move, root=root: root.children[move].N)
                elif move_probs[x][y] == float(0):
                    position.pass_move(mutate=True)
                    return None
                else:
                    # root.children[max_move] = float(0)
                    root.children.pop(max_move)

    # 搜索函数，选择最优分支搜索
    def tree_search(self, root):
        # print("tree search", file=sys.stderr)
        # selection
        chosen_leaf = root.select_leaf()
        # expansion
        try:
            position = chosen_leaf.compute_position()
        except go.IllegalMove:
            position = None
        if position is None:
            # print("illegal move!", file=sys.stderr)
            # See go.Position.play_move for notes on detecting legality
            # 如果搜索出来是非法的行棋步骤，那么砍掉这一个分支并退出操作
            del chosen_leaf.parent.children[chosen_leaf.move]
            return
        # print("Investigating following position:\n%s" % (chosen_leaf.position,), file=sys.stderr)
        # 如果搜索出来的节点分支不是空的，则进行扩展
        move_probs = self.policy_network.run(position)
        for i in range(9):
            for j in range(9):
                if (position.board[i][j] == go.EMPTY):
                    continue
                else:
                    move_probs[i][j] = 0
        chosen_leaf.expand(move_probs)
        # evaluation 对于选择的分支进行估值
        value = self.estimate_value(root, chosen_leaf)
        # backup
        # print("value: %s" % value, file=sys.stderr)
        # 回述这个分支到现在为止的值。
        chosen_leaf.backup_value(value)

    # 对于根结点到现在的选择分支进行估值
    def estimate_value(self, root, chosen_leaf):
        # Estimate value of position using rollout only (for now).
        # (TODO: Value network; average the value estimations from rollout + value network)
        leaf_position = chosen_leaf.position
        current = copy.deepcopy(leaf_position)
        while current.n < self.max_rollout_depth:
            move_probs = self.policy_network.run(current)
            # 寻找一个能够行棋的点
            current = self.play_valid_move(current, move_probs)
            if len(current.recent) > 2 and current.recent[-1].move == current.recent[-2].move == None:
                break
        # else:
            # print("max rollout depth exceeded!", file=sys.stderr)

        # 判断是谁的估值，如果是自己的为正值，如果最后一步节点为对方的为负值
        perspective = 1 if leaf_position.to_play == root.position.to_play else -1
        # 返回这个步骤的当前得分
        return current.score() * perspective

    def play_valid_move(self, position, move_probs):
        for move in sorted_moves(move_probs):
            if go.is_eyeish(position.board, move):
                move_probs[move[0]][move[1]] = float(0)
                continue
            try:
                # 判断当前一个可行的行棋策略，是不是合法的，不合法去寻找下一个点，合法的就返回这个点
                candidate_pos = position.play_move(move, mutate=True)
            except go.IllegalMove:
                continue
            else:
                return candidate_pos
            # 没有找到适合的点的话，就返回pass
        return position.pass_move(mutate=True)
