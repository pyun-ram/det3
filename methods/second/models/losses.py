import torch
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod

REGISTERED_LOSS_CLASSES = {}
def register_loss(cls, name=None):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_LOSS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_LOSS_CLASSES, f"exist class: {REGISTERED_LOSS_CLASSES}"
    REGISTERED_LOSS_CLASSES[name] = cls
    return cls
def get_loss_class(name):
    '''Source: https://github.com/traveller59/second.pytorch'''
    global REGISTERED_LOSS_CLASSES
    assert name in REGISTERED_LOSS_CLASSES, f"available class: {REGISTERED_LOSS_CLASSES}"
    return REGISTERED_LOSS_CLASSES[name]

class Loss(object):
  """
  Abstract base class for loss functions.
  Source: https://github.com/traveller59/second.pytorch
  """
  __metaclass__ = ABCMeta

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    Source: https://github.com/traveller59/second.pytorch
    """
    if ignore_nan_targets:
      target_tensor = torch.where(torch.isnan(target_tensor),
                                prediction_tensor,
                                target_tensor)
    return self._compute_loss(prediction_tensor, target_tensor, **params)

  @abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass

def _softmax_cross_entropy_with_logits(logits, labels):
    """Source: https://github.com/traveller59/second.pytorch"""
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

@register_loss
class WeightedSoftmaxClassificationLoss(Loss):
  """
  Softmax loss function.
  Source: https://github.com/traveller59/second.pytorch
  """

  def __init__(self, logit_scale=1.0, name="WeightedSoftmaxClassificationLoss"):
    """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
    self._logit_scale = logit_scale

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
    num_classes = prediction_tensor.shape[-1]
    prediction_tensor = torch.div(
        prediction_tensor, self._logit_scale)
    per_row_cross_ent = (_softmax_cross_entropy_with_logits(
        labels=target_tensor.view(-1, num_classes),
        logits=prediction_tensor.view(-1, num_classes)))
    return per_row_cross_ent.view(weights.shape) * weights

@register_loss
class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  Source: https://github.com/traveller59/second.pytorch
  """
  def __init__(self, sigma=3.0, code_weights=None, codewise=True, name="WeightedSmoothL1LocalizationLoss"):
    super().__init__()
    self._sigma = sigma
    if code_weights is not None:
      self._code_weights = np.array(code_weights, dtype=np.float32)
      self._code_weights = torch.from_numpy(self._code_weights)
    else:
      self._code_weights = None
    self._codewise = codewise
  def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    diff = prediction_tensor - target_tensor
    if self._code_weights is not None:
      code_weights = self._code_weights.type_as(prediction_tensor).to(target_tensor.device)
      diff = code_weights.view(1, 1, -1) * diff
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
      + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)
    if self._codewise:
      anchorwise_smooth_l1norm = loss
      if weights is not None:
        anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
    else:
      anchorwise_smooth_l1norm = torch.sum(loss, 2)#  * weights
      if weights is not None:
        anchorwise_smooth_l1norm *= weights
    return anchorwise_smooth_l1norm

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  Source: https://github.com/traveller59/second.pytorch
  """
  dense = torch.zeros(size).fill_(default_value)
  dense[indices] = indices_value

  return dense

def _sigmoid_cross_entropy_with_logits(logits, labels):
  # to be compatible with tensorflow, we don't use ignore_idx
  loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
  loss += torch.log1p(torch.exp(-torch.abs(logits)))
  # loss = nn.BCEWithLogitsLoss(reduce="none")(logits, labels.type_as(logits))
  # transpose_param = [0] + [param[-1]] + param[1:-1]
  # logits = logits.permute(*transpose_param)
  # loss_ftor = nn.NLLLoss(reduce=False)
  # loss = loss_ftor(F.logsigmoid(logits), labels)
  return loss

@register_loss
class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25, name="SigmoidFocalClassificationLoss"):
    """Constructor.

    Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
      all_zero_negative: bool. if True, will treat all zero as background.
        else, will treat first label as background. only affect alpha.
    Source: https://github.com/traveller59/second.pytorch
    """
    self._alpha = alpha
    self._gamma = gamma

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    weights = weights.unsqueeze(2)
    if class_indices is not None:
      weights *= indices_to_dense_vector(class_indices,
            prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
    per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = torch.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = torch.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                              (1 - target_tensor) * (1 - self._alpha))

    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    return focal_cross_entropy_loss * weights