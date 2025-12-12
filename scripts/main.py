import argparse
import logging
import os

from label_metric.utils.log_utils import setup_logger
from label_metric.samplers import WeightManager
from label_metric.utils.config_utils import load_config, get_dm, \
    get_model, get_trainer

import os
from dotenv import load_dotenv
load_dotenv()
DATA_DIR_APOCRITA = os.getenv('DATA_DIR_APOCRITA')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    
    if config['where'] == 'eecs':
        raise NotImplementedError("eecs setting is not implemented for config['where'].")
        # dataset_dir = DATA_DIR_EECS
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    elif config['where'] == 'hpc':
        dataset_dir = DATA_DIR_APOCRITA

    logger = logging.getLogger(__name__)
    setup_logger(logger)
    weight_manager = WeightManager(logger, active=config['weight_manager']['active'])
    dm = get_dm(config['data'], dataset_dir, logger, weight_manager)
    dm.setup('fit')
    model = get_model(config['model'], dm, logger, weight_manager)
    trainer = get_trainer(config['trainer'])
    
    trainer.fit(model = model, datamodule = dm)
    trainer.test(model = model, datamodule = dm)
    trainer.predict(model = model, datamodule = dm)

    logger.info(f"exp info: {config['trainer']['name']} {config['trainer']['version']}")
