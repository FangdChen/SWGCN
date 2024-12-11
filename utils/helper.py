import os

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=50):
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value - best_value >= 1e-8) or (expected_order == 'dec' and log_value - best_value <= 1e-8):
        # 当前的结果更优，不应该提前停止
        stopping_step = 0
    else:
        # 当前没有更优结果，继续训练
        stopping_step += 1

    if stopping_step >= flag_step:
        # 当达到人为设置的最大step数时停止
        should_stop = True
    else:
        should_stop = False
    return stopping_step, should_stop
