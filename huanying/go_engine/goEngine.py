# 信息交互模块
import sys
sys.path.append("..")
import go_data.globalData as data


class Engine(object):

    def __init__(self, game_obj):
        self.size = 9
        self.game = game_obj
        self.game.clear()

    def vertex_in_range(self, vertex):
        if vertex == data.PASS:
            return True
        if 1 <= vertex[0] <= self.size and 1 <= vertex[1] <= self.size:
            return True
        else:
            return False

    # 下面是幻影围棋通信接口
    def phantom_play(self, color, vertex):
        """
        幻影围棋落子
        :param color: 落子颜色
        :param vertex: 落子坐标 （x, y) 1-9
        :return: 无
        """
        # 颜色WHITE  / BLACK  vertex : 坐标（x, y）:1-9
        if self.vertex_in_range(vertex):
            if self.game.make_move(color, vertex):
                return
        raise ValueError("illegal move")

    def phantom_move(self, color, step):
        """
        幻影围棋行棋
        :param color: 己方颜色
        :param step: 保存着法的棋步
        :return: 返回是否找到结果
        """
        if color:
            move = self.game.get_move(color)
            step.point.x, step.point.y = move
            if move == data.PASS:
                return False
            else:
                return True
        else:
            raise ValueError("unknown player: {}".format(color))