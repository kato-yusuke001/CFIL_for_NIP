import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    # 日本時間のタイムゾーンを設定
    start_japan = datetime.now()
    formatted_time = start_japan.strftime("%Y-%m-%d_%H-%M-%S")
    
    # ログファイルの設定
    log_filename = f"./logs/log_{formatted_time}.log"

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