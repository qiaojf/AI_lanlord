

import numpy as np
import tensorflow as tf


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity    # for all priority values
        self.tree = np.zeros(2*capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)    # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root

class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6     # [0~1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.002,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self.n_l1 = 128
        self.n_l2 = 256
        self.n_l3 = 512

        
        self._build_net()


        self.memory = Memory(capacity=memory_size)

        if sess is None:
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []
        self.saver = tf.compat.v1.train.Saver() 

    #修改
    def _build_net(self):
        def build_layers(s, c_names, w_initializer, b_initializer):
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, self.n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(s, w1) + b1)
                
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            with tf.compat.v1.variable_scope('l3'):
                w3 = tf.compat.v1.get_variable('w3', [self.n_l2, self.n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.compat.v1.get_variable('b3', [1, self.n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.tanh(tf.matmul(l2, w3) + b3)

            with tf.compat.v1.variable_scope('l4'):
                w4 = tf.compat.v1.get_variable('w4', [self.n_l3, self.n_actions], initializer=w_initializer, collections=c_names)
                b4 = tf.compat.v1.get_variable('b4', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.nn.tanh(tf.matmul(l3, w4) + b4)
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        with tf.compat.v1.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.math.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train'):
            self._train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    #修改
    def choose_action(self, observation, actions,action_space):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            return action_space[action]
        else:
            action = np.random.randint(0, len(actions))
            return actions[action]

    def _replace_target_params(self):
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)])
    
    #修改
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.memory.update(idx, abs_errors[i])
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.cost

    #新增
    def save_model(self, name, win_rate, episode):
        # saver = tf.compat.v1.train.Saver() 
        self.saver.save(self.sess, "Model_pr/"+name+"_"+win_rate+"_"+str(episode)+".ckpt") 
    #新增   
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def load_model(self,model_name):
        # saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess,model_name)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.002,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.compat.v1.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver() 

    def _build_net(self):
        with tf.name_scope('inputs'),tf.compat.v1.variable_scope('inputs', reuse=tf.compat.v1.AUTO_REUSE):
            self.tf_obs = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.compat.v1.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.compat.v1.placeholder(tf.float32, [None, ], name="actions_value")
            
        layer1 = tf.compat.v1.layers.dense(
            inputs=self.tf_obs,
            units=128,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        layer2 = tf.compat.v1.layers.dense(
            inputs=layer1,
            units=256,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        layer3 = tf.compat.v1.layers.dense(
            inputs=layer2,
            units=512,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )
        all_act = tf.compat.v1.layers.dense(
            inputs=layer3,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc4'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'),tf.compat.v1.variable_scope('loss', reuse=tf.compat.v1.AUTO_REUSE):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.math.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'),tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation,actions,action_space):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        actions_ont_hot = np.zeros(len(action_space))
        for i in actions:
            actions_ont_hot[action_space.index(i)] = 1
        a_list = np.array(prob_weights[0]*actions_ont_hot)
        action = np.argmax(a_list)
        return action_space[action]

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self,i_episode):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        if i_episode == 0:
            # saver = tf.compat.v1.train.Saver() 
            self.saver.restore(self.sess,tf.train.latest_checkpoint(r'.\Model_'))
        else:
            # train on episode
            self.sess.run(self.train_op, feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)
        discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_model(self, name, win_rate,episode):
        # saver = tf.compat.v1.train.Saver() 
        self.saver.save(self.sess, "Model_/"+name+"_"+win_rate+'_'+str(episode)+".ckpt")

    def load_model(self,model_name):
        # saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess,model_name)

class PolicysGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.002,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.n_l1 = 128
        self.n_l2 = 256
        self.n_l3 = 512


        self._build_net()

        self.sess = tf.compat.v1.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver() 

    def _build_net(self):
        def build_layers(tf_obs, c_names, w_initializer, b_initializer):
            with tf.compat.v1.variable_scope('l_1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, self.n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(tf_obs, w1) + b1)
                
            with tf.compat.v1.variable_scope('l_2'):
                w2 = tf.compat.v1.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            with tf.compat.v1.variable_scope('l_3'):
                w3 = tf.compat.v1.get_variable('w3', [self.n_l2, self.n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.compat.v1.get_variable('b3', [1, self.n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.tanh(tf.matmul(l2, w3) + b3)

            with tf.compat.v1.variable_scope('l_4'):
                w4 = tf.compat.v1.get_variable('w4', [self.n_l3, self.n_actions], initializer=w_initializer, collections=c_names)
                b4 = tf.compat.v1.get_variable('b4', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.tanh(tf.matmul(l3, w4) + b4)

            self.all_act_prob = tf.nn.softmax(l4, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('inputs'),tf.compat.v1.variable_scope('inputs',):
            self.tf_obs = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.compat.v1.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.compat.v1.placeholder(tf.float32, [None, ], name="actions_value")

        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.tf_obs, c_names, w_initializer, b_initializer)

        with tf.name_scope('loss'),tf.compat.v1.variable_scope('loss',):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.math.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'),tf.compat.v1.variable_scope('train', ):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation,actions,action_space):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        actions_ont_hot = np.zeros(len(action_space))
        for i in actions:
            actions_ont_hot[action_space.index(i)] = 1
        a_list = np.array(prob_weights[0]*actions_ont_hot)
        action = np.argmax(a_list)
        return action_space[action]

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self,i_episode):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        if i_episode == 0:
            saver = tf.compat.v1.train.Saver() 
            saver.restore(self.sess,tf.train.latest_checkpoint(r'.\Model_'))
        else:
            # train on episode
            self.sess.run(self.train_op, feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)
        discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_model(self, name, win_rate,episode):
        # saver = tf.compat.v1.train.Saver() 
        self.saver.save(self.sess, "Model_/"+name+"_"+win_rate+'_'+str(episode)+".ckpt")

    def load_model(self,model_name):
        # saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess,model_name)











































