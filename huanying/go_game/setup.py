import sys
import os
sys.path.append("..")
import go_game.exchange as ex
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# 创建Logger
logger = logging.getLogger("phantom_go")
logger.setLevel(logging.DEBUG)

# 创建Handler

# 终端Handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# 文件Handler
fileHandler = logging.FileHandler('../logs/log17.log', mode='a', encoding='UTF-8')
fileHandler.setLevel(logging.NOTSET)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# 添加到Logger中
# logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

# 打印日志
# logger.debug('debug 信息')
# logger.info('info 信息')
# logger.warning('warn 信息')
# logger.error('error 信息')
# logger.critical('critical 信息')
# logger.debug('%s 是自定义信息' % '这些东西')


ex.run("policy", read_file="../go_data/model/savedmodel")
