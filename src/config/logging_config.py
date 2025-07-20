from loguru import logger

def setup_logging():
    try:
        from tqdm import tqdm
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass
