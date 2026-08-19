from ..utils.logging_utils import get_logger
from ..config.settings import BaseConfig

from ..providers.chat_gateway import ChatGateway
from .chat_base import BaseChatModel


logger = get_logger(__name__)


def get_chat_model_class(config: BaseConfig):
    return ChatGateway.from_experiment_config(config)
    
