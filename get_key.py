import threading, time
from msvcrt import getch

key = "lol"

should_shutdown = False

def thread1():
    global key
    global should_shutdown
    lock = threading.Lock()
    while True:
        with lock:
            key = getch()
        if key in (b' ', b'\x1b'):
            should_shutdown = True
            break

key_listener = threading.Thread(target = thread1)
key_listener.start()

while True:
    time.sleep(1)
    print(key)
    if should_shutdown:
        break

#key_listener.join()
#raise SystemExit()


