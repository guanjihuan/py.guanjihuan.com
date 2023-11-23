# 获取当前日期字符串
def get_date(bar=True):
    import datetime
    datetime_date = str(datetime.date.today())
    if bar==False:
        datetime_date = datetime_date.replace('-', '')
    return datetime_date

# 获取当前时间字符串
def get_time(colon=True):
    import datetime
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    if colon==False:
        datetime_time = datetime_time.replace(':', '')
    return datetime_time

# 获取MAC地址
def get_mac_address():
    import uuid
    mac_address = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    mac_address = '-'.join([mac_address[i:i+2] for i in range(0, 11, 2)])
    return mac_address

# 获取函数的源码
def get_function_source(function_name):
    import inspect
    function_source = inspect.getsource(function_name)
    return function_source

# 获取调用本函数的函数名
def get_calling_function_name(layer=1):
    import inspect
    caller = inspect.stack()[layer]
    calling_function_name = caller.function
    return calling_function_name

# Guan软件包的使用统计（不涉及到用户的个人数据）
global_variable_of_first_guan_package_calling = []
def statistics_of_guan_package():
    import guan
    function_name = guan.get_calling_function_name(layer=2)
    global global_variable_of_first_guan_package_calling
    if function_name not in global_variable_of_first_guan_package_calling:
        global_variable_of_first_guan_package_calling.append(function_name)
        function_calling_name = guan.get_calling_function_name(layer=3)
        if function_calling_name == '<module>':
            try:
                import socket
                datetime_date = guan.get_date()
                datetime_time = guan.get_time()
                current_version = guan.get_current_version('guan')
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(0.5)
                client_socket.connect(('socket.guanjihuan.com', 12345))
                mac_address = guan.get_mac_address()
                message = {
                    'server': 'py.guanjihuan.com',
                    'date': datetime_date,
                    'time': datetime_time,
                    'version': current_version,
                    'MAC_address': mac_address,
                    'function_name': function_name
                }
                import json
                send_message = json.dumps(message)
                client_socket.send(send_message.encode())
                client_socket.close()
            except:
                pass

# 获取Python软件包的最新版本
def get_latest_version(package_name='guan', timeout=2):
    import requests
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=timeout)
    except:
        return None
    if response.status_code == 200:
        data = response.json()
        latest_version = data["info"]["version"]
        return latest_version
    else:
        return None

# 获取软件包的本机版本
def get_current_version(package_name='guan'):
    import importlib.metadata
    try:
        current_version = importlib.metadata.version(package_name)
        return current_version
    except:
        return None

# Guan软件包升级提示
def notification_of_upgrade():
    import random
    rand_number = random.randint(1, 10)
    if rand_number == 5:
        try:
            latest_version = get_latest_version(package_name='guan', timeout=2)
            current_version = get_current_version('guan')
            if latest_version != None and current_version != None:
                if latest_version != current_version:
                    print('提示：您当前使用的版本是 guan-'+current_version+'，目前已经有最新版本 guan-'+latest_version+'。您可以通过以下命令对软件包进行升级：pip install --upgrade guan')
        except:
            pass
notification_of_upgrade()