CONFIG = {
    "udp": {
        "host": "0.0.0.0",
        "port": 5015,
    },
    "tcp": {
        "host": "127.0.0.1",
        "port": 54321,
    },
    "broadcast": {
        "host": "192.168.1.110",
        "port": 4032,
    },
    "host": {
        "host": "192.168.177.64",
        "port": 8010,
    },
    "ros": {
        "namespace": "/my_namespace",
        "cam_topic":"/camera_f/color/image_raw"
    },

}


def get_config(section):
    """通过section名称加载指定配置"""
    if section in CONFIG:
        return CONFIG[section]
    else:
        raise KeyError(f"Configuration section '{section}' not found.")
