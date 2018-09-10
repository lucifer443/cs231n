import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for row in range(num_train):
    exp_score = np.exp(np.matmul(X[row], W))
    sum_exp = np.sum(exp_score)
    softmax_score = exp_score/sum_exp
    loss += -np.log(softmax_score[y[row]])
    for c in range(num_class):
      if c == y[row]:
        dW[:, c] += (exp_score[c]/sum_exp - 1) * X[row]
        continue
      dW[:, c] += exp_score[c]/sum_exp * X[row]
  dW /= num_train
  dW += 2 * reg * W
  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  exp_scores = np.exp(np.matmul(X, W))
  sum_exp = np.sum(exp_scores, axis=1)
  softmax_scores = exp_scores/sum_exp[:, np.newaxis]
  loss += np.sum(-np.log(softmax_scores[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W * W)

  combination = np.copy(softmax_scores)
  combination[np.arange(num_train), y] -= 1
  dW += np.matmul(X.T, combination)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

