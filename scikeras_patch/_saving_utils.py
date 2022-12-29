import os
import tarfile
import tempfile

from io import BytesIO
from types import MethodType
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple

import numpy as np
import tensorflow.keras as keras

from tensorflow import Variable
from tensorflow import io as tf_io
from tensorflow.keras.models import load_model



from keras.utils import losses_utils
from keras.losses import LossFunctionWrapper



######################################################################
##                          CUSTOM LOSSES
######################################################################



###              Epsilon-Insensitive


def e_insensitive( y_true, y_pred, epsilon = 1.):
    
    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    epsilon = tf.cast(epsilon, dtype=K.floatx())
    
    abs_difference = tf.abs(tf.subtract(y_pred, y_true))
    
    return tf.where(abs_difference > epsilon, abs_difference - epsilon , 0)



class EpsilonInsensitiveLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    
    
    
        Standalone usage:
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        # Using 'auto'/'sum_over_batch_size' reduction type.
        h = EpsilonInsensitiveLoss(epsilon = 1.)
        h(y_true, y_pred).numpy()
        >>0.0
        
        
        # Using 'sum' reduction type.
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        h = EpsilonInsensitiveLoss(epsilon = 0.05,reduction = reduction=tf.keras.losses.Reduction.SUM)
        h(y_true, y_pred).numpy()
        >>1.0

        # Using 'None' reduction type.
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        h = EpsilonInsensitiveLoss(epsilon = 0.05,reduction = tf.keras.losses.Reduction.NONE)
        h(y_true, y_pred).numpy()
        >> array([0.55, 0.45], dtype=float32)


    '''
    
    def __init__(self, 
                 epsilon=1.,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'e_insensitive',
                ):
        
        super(EpsilonInsensitiveLoss, self).__init__(e_insensitive,
                                                     epsilon=epsilon,
                                                     name = name,
                                                     reduction=reduction
                                                    )
    def __repr__ (self):
        return 'e_insensitive_epsilon' 

###----------------------PseudoHuber------------------------------

def pseudohuber( y_true, y_pred, delta = 1.):
    
    assert delta != 0., 'Error, delta cannot be 0, division by 0 not allowed'
    
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    delta = tf.cast(delta, dtype=backend.floatx()) 
    
    num = tf.math.square( tf.abs( tf.subtract(y_pred, y_true) ) )
    delta_squared = tf.math.square( delta )
    second_op = tf.divide(num,delta_squared) 
    return delta * tf.math.sqrt( tf.convert_to_tensor(1, dtype=second_op.dtype) + second_op )


class PseudohuberLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    

    '''
    
    def __init__(self, 
                 delta=1.,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'pseudohuber',
                ):
        
        super(PseudohuberLoss, self).__init__(  pseudohuber,
                                                delta=delta,
                                                name = name,
                                                reduction=reduction
                                                )
    def __repr__ (self):
        return 'pseudohuber' 

###----------------------Quantile------------------------------------

def pinball( y_true, y_pred, cuantile = 0.5):
    
    assert (not cuantile < 0.) or (not cuantile > 1.), 'Error, cuantile must be a value between 0 and 1.'
    
    y_pred = tf.cast(y_pred, dtype=K.floatx())
    y_true = tf.cast(y_true, dtype=K.floatx())
    cuantile = tf.cast(cuantile, dtype=K.floatx()) 
    
    diff = tf.subtract(y_pred, y_true) 
    zeros = tf.zeros_like(diff)  
    
    return (1-cuantile) * tf.math.maximum(-diff,zeros )   + cuantile *  tf.math.maximum(zeros,diff )


class QuantileLoss(LossFunctionWrapper):
    '''Computes epsilon-insensitive loss between `y_true` and `y_pred`
    

    '''
    
    def __init__(self, 
                 cuantile=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name = 'pinball',
                ):
        
        super(QuantileLoss, self).__init__(     pinball,
                                                cuantile=cuantile,
                                                name = name,
                                                reduction=reduction
                                                )

    def __repr__ (self):
        return 'quantile' 

#---------------------------------------------------------------------------------------


def _get_temp_folder() -> str:
    if os.name == "nt":
        # the RAM-based filesystem is not fully supported on
        # Windows yet, we save to a temp folder on disk instead
        return tempfile.mkdtemp()
    else:
        return f"ram://{tempfile.mkdtemp()}"


def _temp_create_all_weights(
    self: keras.optimizers.Optimizer, var_list: Iterable[Variable]
):
    """A hack to restore weights in optimizers that use slots.

    See https://tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#slots_2
    """
    self._create_all_weights_orig(var_list)
    try:
        self.set_weights(self._restored_weights)
    except ValueError:
        # Weights don't match, eg. when optimizer was pickled before any training
        # or a completely new dataset is being used right after pickling
        pass
    delattr(self, "_restored_weights")
    self._create_all_weights = self._create_all_weights_orig


def _restore_optimizer_weights(
    optimizer: keras.optimizers.Optimizer, weights: List[np.ndarray]
) -> None:
    optimizer._restored_weights = weights
    optimizer._create_all_weights_orig = optimizer._create_all_weights
    # MethodType is used to "bind" the _temp_create_all_weights method
    # to the "live" optimizer object
    optimizer._create_all_weights = MethodType(_temp_create_all_weights, optimizer)


def unpack_keras_model(
    packed_keras_model: np.ndarray, optimizer_weights: List[np.ndarray]
):
    """Reconstruct a model from the result of __reduce__"""
    temp_dir = _get_temp_folder()
    b = BytesIO(packed_keras_model)
    with tarfile.open(fileobj=b, mode="r") as archive:
        for fname in archive.getnames():
            dest = os.path.join(temp_dir, fname)
            tf_io.gfile.makedirs(os.path.dirname(dest))
            with tf_io.gfile.GFile(dest, "wb") as f:
                f.write(archive.extractfile(fname).read())
    #model: keras.Model = load_model(temp_dir)
    model: keras.Model = load_model(temp_dir,  custom_objects={"EpsilonInsensitiveLoss": EpsilonInsensitiveLoss,
                                                                "PseudohuberLoss" : PseudohuberLoss,
                                                                "QuantileLoss" : QuantileLoss,
                                                                })

    for root, _, filenames in tf_io.gfile.walk(temp_dir):
        for filename in filenames:
            if filename.startswith("ram://"):
                # Currently, tf.io.gfile.walk returns
                # the entire path for the ram:// filesystem
                dest = filename
            else:
                dest = os.path.join(root, filename)
            tf_io.gfile.remove(dest)
    if model.optimizer is not None:
        _restore_optimizer_weights(model.optimizer, optimizer_weights)
    return model


def pack_keras_model(
    model: keras.Model,
) -> Tuple[
    Callable[[np.ndarray, List[np.ndarray]], keras.Model],
    Tuple[np.ndarray, List[np.ndarray]],
]:
    """Support for Pythons's Pickle protocol."""
    temp_dir = _get_temp_folder()
    model.save(temp_dir)
    b = BytesIO()
    with tarfile.open(fileobj=b, mode="w") as archive:
        for root, _, filenames in tf_io.gfile.walk(temp_dir):
            for filename in filenames:
                if filename.startswith("ram://"):
                    # Currently, tf.io.gfile.walk returns
                    # the entire path for the ram:// filesystem
                    dest = filename
                else:
                    dest = os.path.join(root, filename)
                with tf_io.gfile.GFile(dest, "rb") as f:
                    info = tarfile.TarInfo(name=os.path.relpath(dest, temp_dir))
                    info.size = f.size()
                    archive.addfile(tarinfo=info, fileobj=f)
                tf_io.gfile.remove(dest)
    b.seek(0)
    optimizer_weights = None
    if model.optimizer is not None:
        optimizer_weights = model.optimizer.get_weights()
    model_bytes = np.asarray(memoryview(b.read()))
    return (
        unpack_keras_model,
        (model_bytes, optimizer_weights),
    )


def deepcopy_model(model: keras.Model, memo: Dict[Hashable, Any]) -> keras.Model:
    _, (model_bytes, optimizer_weights) = pack_keras_model(model)
    new_model = unpack_keras_model(model_bytes, optimizer_weights)
    memo[model] = new_model
    return new_model


def unpack_keras_optimizer(
    opt_serialized: Dict[str, Any], weights: List[np.ndarray]
) -> keras.optimizers.Optimizer:
    """Reconstruct optimizer."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.deserialize(opt_serialized)
    _restore_optimizer_weights(optimizer, weights)
    return optimizer


def pack_keras_optimizer(
    optimizer: keras.optimizers.Optimizer,
) -> Tuple[
    Callable[
        [Dict[str, Any], keras.optimizers.Optimizer],
        Tuple[Dict[str, Any], List[np.ndarray]],
    ],
]:
    """Support for Pythons's Pickle protocol in Keras Optimizers."""
    opt_serialized = keras.optimizers.serialize(optimizer)
    weights = optimizer.get_weights()
    return unpack_keras_optimizer, (opt_serialized, weights)


def unpack_keras_metric(metric_serialized: Dict[str, Any]) -> keras.metrics.Metric:
    """Reconstruct metric."""
    metric: keras.metrics.Metric = keras.metrics.deserialize(metric_serialized)
    return metric


def pack_keras_metric(
    metric: keras.metrics.Metric,
) -> Tuple[Callable[[Dict[str, Any], keras.metrics.Metric], Tuple[Dict[str, Any]]]]:
    """Support for Pythons's Pickle protocol in Keras Metrics."""
    metric_serialized = keras.metrics.serialize(metric)
    return unpack_keras_metric, (metric_serialized,)


def unpack_keras_loss(loss_serialized):
    """Reconstruct loss."""
    loss: keras.losses.Loss = keras.losses.deserialize(loss_serialized)
    return loss


def pack_keras_loss(loss: keras.losses.Loss):
    """Support for Pythons's Pickle protocol in Keras Losses."""
    loss_serialized = keras.losses.serialize(loss)
    return unpack_keras_loss, (loss_serialized,)
