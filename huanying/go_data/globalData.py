# 全局数据存放
WHITE = -1#棋盘上白子为-1
BLACK = +1#棋盘上黑子为1
EMPTY = 0#棋盘上为空则为0
BORDER = 3#棋盘上边界处表示为3
PASS = (0, 0)#pass表示为（0,0）
RESIGN = "resign"
# 己方棋子颜色
chessColor = None
# 对方棋子颜色
computerSide = None
# 棋盘的点
#二维列表，列表里面为一个10个0的列表，总共有10个相同的子列表
board = [[0 for i in range(11)]for i in range(11)]
#双重循环的表示每个棋子上的坐标位置,并将棋盘上的点都赋值为0，即为空(二维数组）
knownNum = [0 for i in range(3)]
#一维数组[0,0,0]，暂时不知道其作用
roundNum = 0
#初始化点的坐标？
class Point:
    x = -1
    y = -1

#？
class Step:
    point = Point#将Point对象赋值给point变量属性
    value = 0



