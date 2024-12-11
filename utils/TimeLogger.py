import datetime
import logging.handlers
from utils.parser import args
import time

logmsg = ''
timemark = dict()
saveDefault = False
def init_log(start_time, model_path):
    global logger
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    log_dir = model_path
    logging.basicConfig(level=logging.INFO, filename=f'{log_dir}/{start_time}.log', filemode='a')

def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save != None:
        if save:
            logmsg += tem + '\n'
            logger.info(tem)
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)


def marktime(marker):
    global timemark
    timemark[marker] = datetime.datetime.now()


def SpentTime(marker):
    global timemark
    if marker not in timemark:
        msg = 'LOGGER ERROR, marker', marker, ' not found'
        tem = '%s: %s' % (time, msg)
        print(tem)
        return False
    return datetime.datetime.now() - timemark[marker]


def SpentTooLong(marker, day=0, hour=0, minute=0, second=0):
    global timemark
    if marker not in timemark:
        msg = 'LOGGER ERROR, marker', marker, ' not found'
        tem = '%s: %s' % (time, msg)
        print(tem)
        return False
    return datetime.datetime.now() - timemark[marker] >= datetime.timedelta(days=day, hours=hour, minutes=minute,
                                                                            seconds=second)


if __name__ == '__main__':
    log('Steps 1/100: hit = 0.111, ndcg = 0.111          ', save=True, oneline=True)
