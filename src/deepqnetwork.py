import deep_q_graph as dqg
import utils as ut

import tensorflow as tf
import numpy as np
import os
import logging
import mem_top

logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, num_actions, args):
    # remember parameters
    self.num_actions = num_actions
    self.batch_size = args.batch_size
    self.discount_rate = args.discount_rate
    self.history_length = args.history_length
    self.screen_dim = (args.screen_height, args.screen_width)
    self.clip_error = args.clip_error
    self.min_reward = args.min_reward
    self.max_reward = args.max_reward
    self.batch_norm = args.batch_norm

    #start tensorflow session
    self.sess = tf.InteractiveSession()


    # prepare tensors once and reuse them
    self.pre_input = tf.placeholder(tf.uint8, shape=[None, self.screen_dim[1], self.screen_dim[0], self.history_length])
    self.input = tf.to_float(self.pre_input) / 255.

    self.targets = tf.placeholder(tf.float32, shape=[None, self.num_actions])
    self.predqvals = tf.placeholder(tf.float32, shape=[None, self.num_actions])

    self.model = dqg.DeepQArchitecture(self.history_length, self.num_actions, self.input)

    self.cost = tf.nn.l2_loss(self.targets - self.predqvals) #tf.clip_by_value(x, -self.clip_error, self.clip_error)

    if args.optimizer == 'rmsprop':
      self.optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=args.decay_rate)
    elif args.optimizer == 'adam':
      self.optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate)
    elif args.optimizer == 'adadelta':
      self.optimizer = tf.train.AdadeltaOptimizer(rho=args.decay_rate)
    else:
      assert False, "Unknown optimizer"

    deltas = self.optimizer.compute_gradients(self.cost)
    clipped_deltas = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in deltas]
    self.training = self.optimizer.apply_gradients(clipped_deltas)
    # create target model
    self.train_iterations = 0

    self.target_model = dqg.DeepQArchitecture(self.history_length, self.num_actions, self.input)

    self.callback = None

    # setup saver/loader
    self.saver = tf.train.Saver()

    init_op = ut.initialize_all_variables()
    self.sess.run(init_op)

  def update_target_network(self):
      self.target_model.copy_weights(self.model)

  def train(self, minibatch, epoch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    # feed-forward pass for poststates to get Q-values
    feed_poststates = np.transpose(poststates, [0,2,3,1])
    postq = self.sess.run(self.target_model.q_values, {self.pre_input:feed_poststates})
    assert postq.shape == (self.batch_size, self.num_actions)

    # calculate max Q-value for each poststate
    maxpostq = np.amax(postq, axis=1)
    assert len(maxpostq) == self.batch_size

    # feed-forward pass for prestates
    feed_prestates = np.transpose(prestates, [0, 2, 3, 1])
    preq = self.sess.run(self.model.q_values, {self.pre_input:feed_prestates})
    assert preq.shape == (self.batch_size, self.num_actions)

    # make copy of prestate Q-values as targets
    targets = preq.copy()

    # clip rewards between -1 and 1
    rewards = np.clip(rewards, self.min_reward, self.max_reward)

    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        targets[i,action] = float(rewards[i])
      else:
        targets[i,action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

    # perform training
    self.sess.run(self.training, {self.predqvals:preq, self.targets:targets})

    if(self.train_iterations % 100 == 0):
        logger.info("I am actually training!")

    # increase number of weight updates (needed for stats callback)
    self.train_iterations += 1

    # calculate statistics
    if self.callback:
      self.callback.on_train(0)

  def predict(self, states):
    # minibatch is full size, because Neon doesn't let change the minibatch size
    assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

    feed_states = np.transpose(states, [0,2,3,1])
    # calculate Q-values for the states
    qvalues = self.sess.run(self.model.q_values, {self.pre_input:feed_states})
    assert qvalues.shape == (self.batch_size, self.num_actions)
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Q-values: " + str(qvalues[0,:]))

    # transpose the result, so that batch size is first dimension
    return qvalues

  def load_weights(self, load_path):
      self.saver.restore(self.sess, load_path)

  def save_weights(self, save_path):
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    self.saver.save(self.sess, save_path)
