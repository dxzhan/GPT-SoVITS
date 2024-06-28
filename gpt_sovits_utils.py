import os,platform,psutil,signal
'''
    获得系统名称
'''
def get_system():
    return platform.system()
'''
    linux杀死进程函数
'''
def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

'''
    杀死进程方法，自动判断linux或windows
'''
def kill_process(pid):
    if(get_system() == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)