'''
A board is a NxN numpy array.
A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).
A PlayerMove is a (Color, Move) tuple

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
'''
from collections import namedtuple
import copy
import itertools

import numpy as np
import logging
logger = logging.getLogger("phantom_go")#调用日志

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1
# -1    0       1      2    3      4
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)
KO_TIMES = 0  # 打劫，可以反复提子几次。标准规则不能立刻提回来，顾默认为0

#函数链接：https://blog.csdn.net/ben1122334/article/details/102303946
#PlayerMove该函数没有什么作用
class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    pass

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

#合法的移动
class IllegalMove(Exception):

    #默认的函数，构造函数
    def __init__(self, group=None, move=None):
        self.illegal_group = group
        self.last_move = move
    #获得illegal_group的值
    def get_illegal_group(self):
        return self.illegal_group
    #获得last_move的值
    def get_last_move(self):
        return self.last_move

# these are initialized by set_board_size
N = None
ALL_COORDS = []#同等物,一维数组，用来存储棋盘上所有的点的坐标
EMPTY_BOARD = None #设置一个nXn的空棋盘，是一个二维列表，每一行为一个n个0的列表
NEIGHBORS = {}#邻居节点，相连的点，十字形  neighbors,数组表示
DIAGONALS = {}#对角节点，斜线上的点，X字形  diagonals，数组表示


def set_board_size(n):
    """
    Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.
    Also, never do "from go import N, W, ALL_COORDS, EMPTY_BOARD".
    """
    #global的用法：https://blog.csdn.net/weixin_40894921/article/details/88528159
    #定义在函数内部，将对全局变量进行修改其值
    global N, ALL_COORDS, EMPTY_BOARD, NEIGHBORS, DIAGONALS
    if N == n:
        return #N开始时等于None
    N = n
    #创建一个棋盘的一维数组，(0,0)(0,1)...(0,n-1),(1,0）....(n-1,n-1)
    ALL_COORDS = [(i, j) for i in range(n) for j in range(n)]

    #设置一个nXn的空棋盘，是一个二维列表，每一行为一个n个0的列表
    #  numpy的zeros函数：https://blog.csdn.net/DocStorm/article/details/58599124
    EMPTY_BOARD = np.zeros([n, n], dtype=np.int8)

    #检查边界情况，改下棋点的位置情况，来判断邻接点和对角线点的位置
    def check_bounds(c):
        return c[0] % n == c[0] and c[1] % n == c[1]

    #获得neighbors 和 diagonals的 结点位置，用一个数组表示
    NEIGHBORS = {(x, y): list(filter(check_bounds, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
    DIAGONALS = {(x, y): list(filter(check_bounds, [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}

#放置棋子，并在指定该地方的颜色，stones棋子的位置？？？
def place_stones(board, color, stones):
    for s in stones:
        board[s] = color

#发现能到达的位置
def find_reached(board, c):
    #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。数组{}表示
    #c应该是一个位置信息。PS：（1,1）
    color = board[c]
    chain = set([c])#数组列表{}
    reached = set()#数组{}
    frontier = [c]#前沿
    while frontier:
        current = frontier.pop()
        chain.add(current)#加入点C
        for n in NEIGHBORS[current]:#n为该点c的邻接点
            if board[n] == color and not n in chain:#如果邻接点同色并且不是c点，则扩充棋子的范围
                frontier.append(n)
            elif board[n] != color:#邻接点不同色或者为空，已到达的点位置数加一？？？
                reached.add(n)#数组的方法扩展元素
    return chain, reached


# 查找提子周围的点
def find_take_stones_around_points(points, color):
    around_points = set()#数组{}，points是点的坐标
    for point in points:
        for around in NEIGHBORS[point]:
            around_points.add(around)
    for point in points:
        if around_points.__contains__(point):
            #如果包含相同的点，那么就去除该点
            around_points.remove(point)
    #返回列表数组[{  }]
    return list(around_points)

#下棋的顺序，翻转颜色进行下棋
def flip_color(color):
    if color == BLACK:
        return WHITE
    elif color == WHITE:
        return BLACK
    else:
        return EMPTY

#吃棋的规则
#board[n]是一种颜色,c是一个坐标，返回一个颜色
def is_koish(board, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if board[c] != EMPTY: return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    #数组列表，是一种颜色WHITE   BLACK  EMPTY，c点后面只有一种颜色，所以就可以吃棋
    if len(neighbors) == 1 and not EMPTY in neighbors:
        #not EMPTY in neighbors，没有空的棋盘，只有一个元素
        return list(neighbors)[0]
    else:
        return None

#眼，如果是眼那么就返回颜色
def is_eyeish(board, c):
    'Check if c is an eye, for the purpose of restricting（限制） MC rollouts.'
    color = is_koish(board, c)#判断该点是不是吃棋位置，并且获得该点眼的颜色
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[c]#对角线上的点，用列表表示
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if not board[d] in (color, EMPTY):
            diagonal_faults += 1
    if diagonal_faults > 1:
        return None
    else:
        return color


#namedtuple函数链接：https://blog.csdn.net/ben1122334/article/details/102303946
#判断是不是一个整体，这范围内的棋子
class Group(namedtuple('Group', ['id', 'stones', 'liberties', 'color'])):
    """
    stones: a set of Coordinates（坐标） belonging to this group
    liberties: a set of Coordinates that are empty and adjacent（相邻） to this group.
    color: color of this group
    """
    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and self.color == other.color

#自由跟踪器
class LibertyTracker():
    @staticmethod
    def from_board(board):
        board = np.copy(board) #复制棋盘并将其赋值给board

        curr_group_id = 0
        lib_tracker = LibertyTracker()#递归？？     对象
        for color in (WHITE, BLACK):#for循环遍历元祖，第一个颜色为WHITE 第二个为BLACK
            while color in board:#遍历整个棋盘
                curr_group_id += 1
                found_color = np.where(board == color)
                # np.where 的用法https://www.cnblogs.com/massquantity/p/8908859.html
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                #获得能够到达的点的坐标和不能够到达的点或者棋盘上为空的坐标，并用chain和reached保存
                liberties = set(r for r in reached if board[r] == EMPTY)#数组表示{}，表示棋盘上为空的点
                new_group = Group(curr_group_id, chain, liberties, color)#对象
                ##true或者false，判断是不是一个群体（错）
                lib_tracker.groups[curr_group_id] = new_group#是字典类型的变量
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                place_stones(board, FILL, chain)
                #def place_stones(board, color, stones): for s in stones: board[s] = color
                ## 并将相邻的位置填充color为FILL

        lib_tracker.max_group_id = curr_group_id

        # 设置一个nXn的空棋盘，是一个二维列表，每一行为一个n个0的列表
        #  numpy的zeros函数：https://blog.csdn.net/DocStorm/article/details/58599124
        liberty_counts = np.zeros([N, N], dtype=np.uint8)

        for group in lib_tracker.groups.values():#遍历字典的值，是一个Group得对象实例
            num_libs = len(group.liberties)#liberties 数组表示{}，表示棋盘上为空的点，num_lib为点的个数
            for s in group.stones:
                liberty_counts[s] = num_libs
        lib_tracker.liberty_cache = liberty_counts

        return lib_tracker

    def __init__(self, group_index=None, groups=None, liberty_cache=None, max_group_id=1):
        # group_index: a NxN numpy array of group_ids. -1 means no group
        # groups: a dict of group_id to groups
        # liberty_cache: a NxN numpy array of liberty counts
        self.group_index = group_index if group_index is not None else -np.ones([N, N], dtype=np.int16)
        self.groups = groups or {}
        self.liberty_cache = liberty_cache if liberty_cache is not None else np.zeros([N, N], dtype=np.uint8)
        self.max_group_id = max_group_id

    #返回一个LibertyTracker的对象
    def __deepcopy__(self, memodict={}):
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        new_groups = {
            group.id: Group(group.id, set(group.stones), set(group.liberties), group.color)
            for group in self.groups.values()
        }
        return LibertyTracker(new_group_index, new_groups, liberty_cache=new_lib_cache, max_group_id=self.max_group_id)

    def add_stone(self, color, c):
        ## Represents "group not found" in the LibertyTracker object  MISSING_GROUP_ID=-1
        assert self.group_index[c] == MISSING_GROUP_ID
        captured_stones = set()#捕获的棋子？？
        opponent_neighboring_group_ids = set()#敌对的邻居节点
        friendly_neighboring_group_ids = set()#友好的邻居节点
        empty_neighbors = set()#为空的邻居节点

        #判断邻居节点所对应的棋子群的标号，字典类型表示群组编号和群组
        for n in NEIGHBORS[c]:
            neighbor_group_id = self.group_index[n]
            if neighbor_group_id != MISSING_GROUP_ID:
                neighbor_group = self.groups[neighbor_group_id]#是一个Group的对象
                if neighbor_group.color == color:
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                empty_neighbors.add(n)

        new_group = self._create_group(color, c, empty_neighbors)#其实也是一个Group的对象

        for group_id in friendly_neighboring_group_ids:
            new_group = self._merge_groups(group_id, new_group.id)#合并两个相同的颜色且相连的棋组群

        for group_id in opponent_neighboring_group_ids:
            neighbor_group = self.groups[group_id]
            if len(neighbor_group.liberties) == 1:
                captured = self._capture_group(group_id)#返回的是棋组里面的坐标点
                captured_stones.update(captured)
                #数组的更新方法，把captured放到captured_stones里面，且没有重复
            else:
                self._update_liberties(group_id, remove={c})

        self._handle_captures(captured_stones)#更新棋群

        # suicide is illegal
        if len(new_group.liberties) == 0:
            #该棋群没有气了，则会被提子
            raise IllegalMove

        return captured_stones

    def _create_group(self, color, c, liberties):
        self.max_group_id += 1
        new_group = Group(self.max_group_id, set([c]), liberties, color)
        self.groups[new_group.id] = new_group
        self.group_index[c] = new_group.id
        self.liberty_cache[c] = len(liberties)
        return new_group

    #合并两个相同的组群
    def _merge_groups(self, group1_id, group2_id):
        group1 = self.groups[group1_id]#Group的对象
        group2 = self.groups[group2_id]#Group的对象
        group1.stones.update(group2.stones)
        del self.groups[group2_id]
        for s in group2.stones:
            self.group_index[s] = group1_id

        self._update_liberties(group1_id, add=group2.liberties, remove=(group2.stones | group1.stones))

        return group1

    #返回死棋的位置？？？
    def _capture_group(self, group_id):
        dead_group = self.groups[group_id]#Group对象实例
        del self.groups[group_id]
        for s in dead_group.stones:
            self.group_index[s] = MISSING_GROUP_ID
            self.liberty_cache[s] = 0
        return dead_group.stones

    #更新棋组气的数量
    def _update_liberties(self, group_id, add=None, remove=None):
        group = self.groups[group_id]
        if add:
            group.liberties.update(add)
        if remove:
            group.liberties.difference_update(remove)

        new_lib_count = len(group.liberties)
        for s in group.stones:
            self.liberty_cache[s] = new_lib_count

    #更新棋组气的数量
    def _handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                group_id = self.group_index[n]
                if group_id != MISSING_GROUP_ID:
                    self._update_liberties(group_id, add={s})


class Position():
    def __init__(self, board=None, n=0, komi=7.5, caps=(0, 0), lib_tracker=None,
                 ko=None,ko_times=0, recent=tuple(),
                 to_play=BLACK, my_color=BLACK):
        '''
        board: a numpy array
        n: an int representing moves played so far
        komi: a float, representing points given to the second player.
        caps: a (int, int) tuple of captures for B, W.
        lib_tracker: a LibertyTracker object
        ko: a Move
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        to_play: BLACK or WHITE
        '''
        self.board = board if board is not None else np.copy(EMPTY_BOARD)
        self.n = n  # 到现在我方的行棋数
        self.komi = komi  # 贴目数
        self.caps = caps  # 存储黑棋白棋的目数的数组。0为白棋，1为黑棋
        self.lib_tracker = lib_tracker or LibertyTracker.from_board(self.board)#a LibertyTracker object
        self.ko = ko  # 一个劫的眼位，对面的眼
        self.ko_times = ko_times
        self.recent = recent  # 最后一次行棋
        self.to_play = to_play  #我方行棋颜色
        self.my_color = my_color

    #深拷贝，返回一个Position对象实例
    def __deepcopy__(self, memodict={}):
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return Position(new_board, self.n, self.komi, self.caps, new_lib_tracker, self.ko, self.ko_times, self.recent, self.to_play, self.my_color)

    def __str__(self):
        pretty_print_map = {
            WHITE: 'O',
            EMPTY: '.',
            BLACK: 'X',
            FILL: '#',
            KO: '*',
        }
        board = np.copy(self.board)
        captures = self.caps# 存储黑棋白棋的目数的数组。0为白棋，1为黑棋
        if self.ko is not None:#对面的眼不为空
            place_stones(board, KO, [self.ko])#在指定的位置填充颜色“KO”
        raw_board_contents = []
        for i in range(N):
            row = []#棋盘行
            for j in range(N):
                appended = '<' if (self.recent and (i, j) == self.recent[-1].move) else ' '
                row.append(pretty_print_map[board[i,j]] + appended)
            raw_board_contents.append(''.join(row))

        row_labels = ['%2d ' % i for i in range(N, 0, -1)]
        annotated_board_contents = [''.join(r) for r in zip(row_labels, raw_board_contents, row_labels)]
        header_footer_rows = ['   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n, *captures)
        return annotated_board + details

    def set_my_color(self, color):
        self.my_color = color

    #返回一个可以合法移动的棋群
    def get_illegal_move_group(self, illegal_move):
        """
        获取幻影围棋非法猜测对方落子周围的群
        :param illegal_move: 猜测点最后非法位置
        :return: 群
        """
        n = NEIGHBORS[illegal_move][0]#用数组表示，里面是（x，y）
        neighbor_group_id = self.lib_tracker.group_index[n]
        if neighbor_group_id == -1:
            return None
        return self.lib_tracker.groups[neighbor_group_id]

    #自杀式的下棋，move是一个棋盘上点的坐标
    def is_move_suicidal(self, move):
        potential_libs = set()#数组表示，其实是集合{}不能有重复元素
        for n in NEIGHBORS[move]:
            neighbor_group_id = self.lib_tracker.group_index[n]
            if neighbor_group_id == MISSING_GROUP_ID:
                # at least one liberty after playing here, so not a suicide
                #至少有一个口气存在要不然为自杀
                #MISSING_GROUP_ID=-1
                return False
            neighbor_group = self.lib_tracker.groups[neighbor_group_id]
            #返回地neighbor_group_id个棋群，neighbor_group是Group的一个实例
            if neighbor_group.color == self.to_play:
                potential_libs |= neighbor_group.liberties
            elif len(neighbor_group.liberties) == 1:
                # would capture an opponent group if they only had one lib.
                return False
        # it's possible to suicide by connecting several friendly groups
        # each of which had one liberty.
        potential_libs -= set([move])
        #潜在的气，防止自杀行为，利用集合的差集，减少重复的点

        return not potential_libs

    #move是一个棋盘上点的坐标
    def is_move_legal(self, move, color=None):
        """Checks that a move is on an empty space, not on ko（劫眼）, and not suicide"""
        # logger.error("判断行棋是否合法：")
        # logger.error(move)
        # logger.error(color)
        if move is None:
            return True
        if self.board[move] != EMPTY:
            return False
        # 我们是幻影围棋，故在此做规定：我们只对自己的棋子才进行打劫判断
        if move == self.ko and color != (self.my_color*-1):
            # logger.error("进入ko判断，ko_times = ")
            # logger.error(self.ko_times)
            if self.ko_times >= KO_TIMES:
                # self.ko_times = 0
                # self.ko = None
                return False
            else:
                return True
        if self.is_move_suicidal(move):
            return False

        return True

    def pass_move(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        pos.n += 1
        pos.recent += (PlayerMove(pos.to_play, None),)
        pos.to_play *= -1
        pos.ko = None
        return pos

    def flip_playerturn(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        # pos.ko = None
        pos.to_play *= -1
        return pos

    def get_liberties(self):
        return self.lib_tracker.liberty_cache

    def play_move(self, c, color=None, mutate=False):
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)
        if color is None:
            color = self.to_play

        pos = self if mutate else copy.deepcopy(self)

        if c is None:
            pos = pos.pass_move(mutate=mutate)
            return pos

        if not self.is_move_legal(c, color):
            raise IllegalMove(self.get_illegal_move_group(c), c)

        place_stones(pos.board, color, [c])
        captured_stones = pos.lib_tracker.add_stone(color, c)
        place_stones(pos.board, EMPTY, captured_stones)

        opp_color = color * -1

        if c == self.ko and color == self.my_color:
            self.ko_times += 1

        if (len(captured_stones) == 1) and (is_koish(self.board, c) == opp_color) and (color == (-1*self.my_color)):
            new_ko = list(captured_stones)[0]
        else:
            new_ko = self.ko
        # logger.error("new_ko是：ko_times是：")
        # logger.error(new_ko)
        # logger.error(self.ko_times)
        # logger.error("captured_stones的信息为：")
        # logger.error(len(captured_stones))
        # logger.error(is_koish(self.board, c))
        # logger.error(color)
        # logger.error(self.my_color)

        if pos.to_play == BLACK:
            new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
        else:
            new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))

        pos.n += 1
        pos.caps = new_caps
        pos.ko = new_ko
        pos.ko_times = self.ko_times
        pos.recent += (PlayerMove(color, c),)
        pos.to_play *= -1
        return pos

    '''
    对点A落下那粒子的位置进行判断
    1.这个位置是否可以落子
    2.当A落子后,是否能吃掉部分黑棋.让黑棋提掉.(要提掉那些子)
    3.当A落子后,没有想到黑棋没有提掉,倒形成了自提.(如果自提,要提掉那些子)

    判断能否做活、设置两个真眼、做活棋的方法：
    对该群周围全赋值为A颜色
    把该群全部赋值为B颜色
    for i in group
        把i点设置为空点
        for j in group and i !=j
            把j点设置为空点
            然后对这两个点行棋
            当切仅当着两个点行棋都不合法时，返回这两个点
            这样这两个点设置为空，这样此处就是两个真眼活棋了
    '''
    def deal_illegal_move_group(self, points, last_move):
        opp_color = flip_color(self.to_play)
        tmp_pos = copy.deepcopy(self)
        tmp_pos.board[last_move] = opp_color
        for i in range(len(points)):
            tmp_pos.board[points[i]] = EMPTY
            for j in range(len(points)-1):
                tmp_pos.board[points[j]] = EMPTY
                if not tmp_pos.is_move_legal(points[i]) and not tmp_pos.is_move_legal(points[j]):
                    self.board[points[i]] = EMPTY
                    self.board[points[j]] = EMPTY
                    self.board[last_move] = opp_color
                tmp_pos.board[points[j]] = opp_color
            tmp_pos.board[points[i]] = opp_color

    def score(self):
        'Return score from B perspective（正数）. If W is winning, score is negative（负数）.'
        #territory  领土，占位置的面积
        working_board = np.copy(self.board)
        while EMPTY in working_board:
            unassigned_spaces = np.where(working_board == EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            territory, borders = find_reached(working_board, c)
            border_colors = set(working_board[b] for b in borders)
            X_border = BLACK in border_colors
            O_border = WHITE in border_colors
            if X_border and not O_border:
                territory_color = BLACK
            elif O_border and not X_border:
                territory_color = WHITE
            else:
                territory_color = UNKNOWN # dame, or seki
            place_stones(working_board, territory_color, territory)

        return np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi

    def result(self):
        score = self.score()
        if score > 0:
            return 'B+' + '%.1f' % score
        elif score < 0:
            return 'W+' + '%.1f' % abs(score)
        else:
            return 'DRAW'

set_board_size(9)
