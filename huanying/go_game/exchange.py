# 信息交互模块
import sys

sys.path.append("..")
import copy
import os
import logging
import warnings

warnings.filterwarnings("ignore")
import go_data.globalData as data
import go_engine.goEngine as go_engine
import go_engine.go as go
import utils.go_utils as utils
from go_engine.netEngine import PolicyNetWork
from go_engine.aiEngine import PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''信息交互模块

目前幻影围棋存在的大坑还有：
1、被拒绝的点，我们并不知道是对面的棋子，还是对面的眼位
    如果是对面的眼位，那么我们赋值为对面的棋子的话，在我方
    棋盘上会出现对面无气的棋子还在棋盘上，没有被提子的情况
    这会引发异常
2、但是这种情况，是因为我方棋盘信息不完整，而且有些点的位置是赋值错误的
    所以，这样的话，就需要解决对 对方眼位被拒绝的情况

对于非法情况：
    1、可能是我着棋被拒绝，赋值对面的时候才生非法局面：
        解决思路：
        我们有非法落子的异常捕捉功能，在最后一个落子被拒绝的点
        假设这个点为 P(x, y）这样，异常捕获的时候，同样要让我们能够
        接收到产生非法的这个对方棋子的群（即点坐标的集合）。

    2、是我走棋，而且被允许了，但是，我自己以为我可以提子的
    这样我方错误以为提子了，对方的区域被赋值成为了空、
    但是，如果我方继续在这儿走棋，又会被赋值为对方的，最终有可能的话，会剩下对方的眼位，
    不去走了，或者走中敌人的单眼或假眼可以提子的情况。
    所以同样回到了第一类情况的处理。
    暂时考虑第一种情况的直接处理办法。

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

设置一个布尔值，用于检测是否需要进入猜、试棋系统（就是试棋，能提子就提，不能就把这块设置成敌人的活棋）
并且控制进入执行  群的棋子数目为 次数，次数达到了，就做处理
然后设置布尔值为非，这样退出检测行棋


下面先解决非法抛出异常的问题：


目前遇到的问题：
    1、到了后面会提交自己的点
    2、提交自己的点没有抛出异常
    3、这个点被拒绝后，赋值为对方的时候没有抛出非法落子异常，而出现其他类异常

被提子之后，可以完全知道周围的棋子


存在问题：
    系统提交了已知的对方的点的位置，然而我们不应该走，这是非法的！！！！！  捕捉到非法
    按理说不应该吧这个点找出来提交！！！！！


提子信息解决方案：
    目前，经过测试后发现，以前我们直接对棋盘数组，把被提子的点复制为空，这样会导致棋盘
    的群信息记录有遗漏，这样会在后面出现很多bug。当然，这就是信息不正确导致的逻辑错误

    我们最新的解决方方案：
    1、我们准备两个position ,其中一个为：每走一步，都会实时更新的，而另一个为慢一步的position
        意思就是，我们这次走了，被同意了，但是，我们要下一次走了，被同意了，才会把上一步的信息告诉它。
        当我们接收到提子或者被提子的信息的时候，就会让那个满了一步的position发挥作用。

    2、对于对方棋子被提：
        对慢了一步的position，依次把刚才系统告诉我们的提子的点，来进行make_move()操作，当然是走对方的颜色
        的棋子的。然后，我们再把上一步马上进行make_move操作，这样的话，我方就能正确的提子对方了。

    3、对于我方棋子被提：
        我们就找到我方被提的那几个子的周围的几个坐标，然后，先把我方上一步棋补上，再依次把那些坐标用make_move来着棋
        当着完以后，系统自然会把群信息更新，并且把要被题的子踢掉。
        所以，慢一步的那个position只是在我方提对方的子的时候才有用。

    对于这种方案的优点在于：
        本身可以正确的解决提子问题，同时可以通过提子信息，来完善对方棋子的信息，其实就是我方被提子周围的棋子，
        肯定是对面的棋子了啊！！


'''
# 创建Logger， 用于打印日志，在程序入口配置，获取名字天填写一致即可
logger = logging.getLogger("phantom_go")


def run(strategy, read_file=None):
    # 初始化神经网络
    n = PolicyNetWork(use_cpu=True)
    if strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
    else:
        sys.stderr.write("Unknown strategy")
        sys.exit()

    # 传入神经网络，初始化幻影围棋策略网络
    gtp_engine = go_engine.Engine(instance)
    sys.stderr.write("Winter引擎已就绪\n")
    sys.stderr.flush()

    # 设置是否需要进入检测模式，初始为否，正常用策略网络下棋
    is_need_entry_check_mode = False
    # 设置需要检测行棋模式的次数，当次数为0的时候，把上面的布尔值设置为False
    num_least_need_check = 0
    # 需要检测的点的坐标
    need_check_points = None
    last_move = None
    pre_move = None

    step = data.Step  # 生成着棋，调用函数，直接传入该对象，属于传引用
    while True:
        # 清空缓存区（标准输入输出）
        sys.stdout.flush()
        message = input().split(' ')
        sys.stdin.flush()

        # 打印收到的消息日志
        t_str = "信息长度:" + str(len(message))
        logger.info(t_str)
        logger.info(message)

        if is_need_entry_check_mode:
            '''需要进入检测模式下棋'''
            logger.error("在检查模式里面，数目={0}".format(num_least_need_check))
            num_least_need_check -= 1

            if "refuse" in message[0]:
                if num_least_need_check == -1:
                    is_need_entry_check_mode = False
                    # 现在要做的就是把这部分非法的棋子处理，直接将其做两眼成为对方的活棋部分
                    gtp_engine.game.position.deal_illegal_move_group(need_check_points, last_move)
                    if gtp_engine.phantom_move(data.chessColor, step):
                        print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
                    else:
                        print("move pass")
                # group存储的是内部处理的0-8 那个对称坐标，传进board不需要处理，但是向外部使用需要转换出去
                vertex = need_check_points[num_least_need_check]
                step.point.x, step.point.y = utils.unparse_pygtp_coords(vertex)

                print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
            elif "accept" in message[0]:
                num_least_need_check = 0
                is_need_entry_check_mode = False
                try:
                    gtp_engine.game.make_move(data.chessColor, (step.point.x, step.point.y))
                except go.IllegalMove as illegal:
                    logger.error("捕捉到异常，被接受的点存在非法赋值")

        elif "accept" in message[0]:
            '''着法被接受'''
            data.knownNum[data.chessColor] += 1
            data.board[step.point.x][step.point.y] = data.chessColor
            if go.KO_TIMES == 0:
                gtp_engine.game.position.ko = None
            else:
                gtp_engine.game.position.ko_times = 0
            logger.error("被接受的点为：（{0}，{1}）".format(step.point.x, step.point.y))
          #  print("被接受的点为：（{0}，{1}）".format(step.point.x, step.point.y))
            try:
                gtp_engine.game.make_move(data.chessColor, (step.point.x, step.point.y))
            except go.IllegalMove as illegal:
                logger.error("捕捉到异常，被接受的点存在非法赋值")

        elif "refuse" in message[0]:
            '''着法被拒绝'''
            data.board[step.point.x][step.point.y] = data.computerSide
            logger.error("被拒绝的点为：（{0}，{1}）".format(step.point.x, step.point.y))
           # print("被拒绝的点为：（{0}，{1}）".format(step.point.x, step.point.y))
            try:
                gtp_engine.game.make_move(data.computerSide, (step.point.x, step.point.y))
            except go.IllegalMove as illegal:
                logger.error("捕捉到异常，被拒绝的点存在非法赋值")

                need_check_points = list(illegal.illegal_group.stones)
                logger.info("非法群：")
                logger.info(need_check_points)

                is_need_entry_check_mode = True
                # 坐标旋转，从外部传进去使用需要旋转）
                vertex = (step.point.x, step.point.y)
                last_move = utils.parse_pygtp_coords(vertex)

                num_least_need_check = len(need_check_points) - 1
                step.point.x, step.point.y = need_check_points[num_least_need_check]
                logger.error("测试走非法群的一个点：{0},{1}".format(step.point.x, step.point.y))

                print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
                continue
            except Exception as e:
                logger.error("捕捉到异常类：")
                logger.error(e)
                logger.error("此时坐标信息：")
                logger.error(gtp_engine.game.position.board)
                return
            except:
                logger.error("捕捉到其他类异常")

            if gtp_engine.phantom_move(data.chessColor, step):
                print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
                pre_move = (step.point.x, step.point.y)
            else:
                print("move pass")
        elif "move" in message[0]:
            '''该我方行棋：
                此处需要对慢一步的position进行处理。
            '''
            gtp_engine.game.pre_position = copy.deepcopy(gtp_engine.game.position)
            if message[1] == "go":
                data.knownNum[data.computerSide] += 1
            elif message[1] == "pass":
                pass
            if gtp_engine.phantom_move(data.chessColor, step):
                print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
                pre_move = (step.point.x, step.point.y)
            else:
                print("move pass")

        elif "taked" in message[0]:
            '''我方棋子被提'''
            logger.error("我方被提子")
            num = int(message[1])
            position = message[2]
            points = []
            for i in range(num):
                x = ord(position[2 * i]) - ord('A') + 1
                y = ord(position[2 * i + 1]) - ord('A') + 1
                data.board[x][y] = data.EMPTY
                # 外部棋盘坐标需要翻折中心对称过去
                vertex = (x, y)
                c = utils.parse_pygtp_coords(vertex)
                # gtp_engine.game.position.board[c] = data.EMPTY
                points.append(c)
            around_points = go.find_take_stones_around_points(points, data.chessColor)
            logger.error("被提子周围的子为：")
            logger.error(around_points)
            for i in range(2):
                for around in around_points:
                    try:
                        c = utils.unparse_pygtp_coords(around)
                        gtp_engine.game.make_move(data.computerSide, c)
                    except:
                        pass
            logger.error("提子日志：")
            logger.error(gtp_engine.game.position.board)

        elif "take" in message[0]:
            '''我方着棋提对方的子：
                此处需要用我们的双position法来处理。'''
            logger.error("我方提子对方")

            num = int(message[1])
            position = message[2]
            for i in range(num):
                x = ord(position[2 * i]) - ord('A') + 1
                y = ord(position[2 * i + 1]) - ord('A') + 1
                data.board[x][y] = data.EMPTY
                # logger.info("提子数：{0}，坐标为：{1}{2}".format(i,x,y))
                # gtp_engine._game.position.board
                # 外部棋盘坐标需要翻折中心对称过去
                vertex = (x, y)
                # c = utils.parse_pygtp_coords(vertex)
                # gtp_engine.game.position.board[c] = data.EMPTY
                try:
                    gtp_engine.game.pre_make_move(data.computerSide, vertex)
                except:
                    pass
            gtp_engine.game.pre_position.ko = None
            gtp_engine.game.pre_make_move(data.chessColor, pre_move)
            gtp_engine.game.position = copy.deepcopy(gtp_engine.game.pre_position)

            logger.error("提子日志：")
            logger.error(gtp_engine.game.position.board)

        elif "new" in message[0]:
            '''新开棋局'''
            if message[1] == "black":
                data.chessColor = data.BLACK
                data.computerSide = data.WHITE
                gtp_engine.game.position.set_my_color(data.BLACK)
            else:
                data.chessColor = data.WHITE
                data.computerSide = data.BLACK
                gtp_engine.game.position.set_my_color(data.WHITE)
            if data.chessColor == data.BLACK:
                if gtp_engine.phantom_move(data.BLACK, step):
                    print("move {0}{1}".format(chr(step.point.x - 1 + ord('A')), chr(step.point.y - 1 + ord('A'))))
                else:
                    print("move pass")
        elif message[0] == "error":
            '''错误'''
            sys.stdin.flush()
        elif message[0] == "name?":
            '''告诉平台我方名字'''
            sys.stdin.flush()
            print("name Winter")
        elif message[0] == "end":
            '''游戏结束'''
            sys.stdin.flush()
            break
        elif message[0] == "quit":
            '''退出引擎'''
            sys.stdin.flush()
            break
        message.clear()
