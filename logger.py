import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import os

def setup_logger(name, log_file="logs", level=logging.INFO):
    # 日本時間のタイムゾーンを設定
    start_japan = datetime.now()
    formatted_time = start_japan.strftime("%Y-%m-%d_%H-%M-%S")
    
    # ログファイルの設定
    log_dir = f"./{log_file}" 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{name}_log_{formatted_time}.log")

    # ロガーを設定
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ローテーティングハンドラを設定
    handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1)
    handler.suffix = "%Y-%m-%d"
    
    # フォーマッタを設定
    class JSTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    formatter = JSTFormatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    
    # ハンドラをロガーに追加
    logger.addHandler(handler)
    
    return logger


if __name__ == "__main__":
    logger = setup_logger("example", "example")
    logger.info("Hello, World!")    