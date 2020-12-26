###  A Coordinate is a tuple index into the board.
###定式，步法，算法实现
import go_data.globalData as data
import numpy as np

coo = [[(5, 6), (5, 2)],
       [(4, 5), (4, 3), (5, 4)],
       [(2, 3), (2, 5), (3, 6), (3, 3), (4, 3), (3, 5), (2, 4)],
       [(3, 4), (2, 4), (4, 4), (5, 4), (4, 5), (3, 5), (4, 3), (4, 2)]]


def stereotypes(valueBoard, baseBoard):
    print('步数',data.roundNum)
    if data.roundNum == 0:
        # 第一步
        if baseBoard[coo[0][0]] == data.EMPTY:
            valueBoard[coo[0][0]] += 100

        else:
            valueBoard[coo[0][1]] += 100

    if data.roundNum == 1:
        # 第二步
        # 第一步被接受
        if (baseBoard[coo[0][0]] == data.ourColor
            and baseBoard[coo[1][0]] == data.EMPTY):
            valueBoard[coo[1][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][1]] == data.EMPTY):
            valueBoard[coo[1][1]] += 100

        # 第一步没被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.EMPTY):
            valueBoard[coo[1][1]] += 100

        else:
            valueBoard[coo[1][2]] += 100

    if data.roundNum == 2:
        # 第三步
        # 第一二步被接受
        if (baseBoard[coo[0][0]] == data.ourColor
            and baseBoard[coo[1][0]] == data.ourColor
            and baseBoard[coo[2][0]] == data.EMPTY):
            valueBoard[coo[2][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][3]] == data.EMPTY):
            valueBoard[coo[2][3]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][4]] == data.EMPTY):
            valueBoard[coo[2][4]] += 100

        # 第一步被接受，但第二步没被接受
        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][0]] == data.EMPTY):
            valueBoard[coo[2][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][3]] == data.EMPTY):
            valueBoard[coo[2][3]] += 100

        # 第一步没被接受，但第二步被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][1]] == data.EMPTY):
            valueBoard[coo[2][1]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][2]] == data.EMPTY):
            valueBoard[coo[2][2]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][6]] == data.EMPTY):
            valueBoard[coo[2][6]] += 100

        # 第一二步都没被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][1]] == data.EMPTY):
            valueBoard[coo[2][1]] += 100

        else:
            valueBoard[coo[2][5]] += 100

    if data.roundNum == 3:
        # 第一二三步都被接受
        if (baseBoard[coo[0][0]] == data.ourColor
            and baseBoard[coo[1][0]] == data.ourColor
            and baseBoard[coo[2][0]] == data.ourColor
            and baseBoard[coo[3][0]] == data.EMPTY):
            valueBoard[coo[3][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][1]] == data.EMPTY):
            valueBoard[coo[3][1]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][3]] == data.EMPTY):
            valueBoard[coo[3][3]] += 100

        # 第一二步被接受，但第三步没被接受
        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][3]] == data.ourColor
              and baseBoard[coo[3][1]] == data.EMPTY):
            valueBoard[coo[3][1]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][3]] == data.ourColor
              and baseBoard[coo[3][0]] == data.EMPTY):
            valueBoard[coo[3][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][3]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][4]] == data.ourColor
              and baseBoard[coo[3][0]] == data.EMPTY):
            valueBoard[coo[3][0]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][0]] == data.ourColor
              and baseBoard[coo[2][4]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        # 第一三步被接受，但第二步没被接受
        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][6]] == data.EMPTY):
            valueBoard[coo[3][6]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][0]] == data.ourColor
              and baseBoard[coo[3][7]] == data.EMPTY):
            valueBoard[coo[3][7]] += 100

        # 第一步被接受，但第二三步没被接受
        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][3]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][0]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][3]] == data.ourColor
              and baseBoard[coo[3][6]] == data.EMPTY):
            valueBoard[coo[3][6]] += 100

        # 第一步没被接受，但第二三步被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][0]] == data.EMPTY):
            valueBoard[coo[3][0]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][1]] == data.EMPTY):
            valueBoard[coo[3][1]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][3]] == data.EMPTY):
            valueBoard[coo[3][3]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][6]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][6]] == data.ourColor
              and baseBoard[coo[3][4]] == data.EMPTY):
            valueBoard[coo[3][4]] += 100

        # 第一二步没被接受，但第三步被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][4]] == data.EMPTY):
            valueBoard[coo[3][4]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][2]] == data.EMPTY):
            valueBoard[coo[3][2]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][5]] == data.ourColor
              and baseBoard[coo[3][0]] == data.EMPTY):
            valueBoard[coo[3][0]] += 100

        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][2]] == data.ourColor
              and baseBoard[coo[2][1]] == data.ourColor
              and baseBoard[coo[3][4]] == data.EMPTY):
            valueBoard[coo[3][4]] += 100

        # 第一三步没被接受，但第二步被接受
        elif (baseBoard[coo[0][1]] == data.ourColor
              and baseBoard[coo[1][1]] == data.ourColor
              and baseBoard[coo[2][2]] == data.ourColor
              and baseBoard[coo[3][5]] == data.EMPTY):
            valueBoard[coo[3][5]] += 100

        else:
            valueBoard[coo[3][4]] += 100
    print(valueBoard)
    return valueBoard

