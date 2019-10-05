from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.utils import logger

log = logger.get()

MODEL_REGISTRY = {}


def RegisterModel(model_name):
  """Registers a model class"""

  def decorator(f):
    MODEL_REGISTRY[model_name] = f
    return f

  return decorator


def get_model(model_name, *args, **kwargs):
  log.info("Model {}".format(model_name))
  if model_name in MODEL_REGISTRY:
    return MODEL_REGISTRY[model_name](*args, **kwargs)
  else:
    raise ValueError("Model class does not exist {}".format(model_name))
