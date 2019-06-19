import numpy as np
from copy import deepcopy
from hac.experience_buffer import ExperienceBuffer
from hac.actor import Actor
from hac.critic import Critic


class Layer:
    """Base layer object for multi-layer hierarchical policies.

    TODO

    Attributes
    ----------
    layer_number : int
        the level of the layer (0 being the lowest)
    flags : argparse.Namespace
        the parsed arguments from the command line (see options.py)
    sess : tf.Session
        the tensorflow session
    time_limit : int
        maximum number of actions that can be performed by a given layer
    current_state : array_like
        the most recent observation
    goal : array_like
        the most recent goal array
    buffer_size_ceiling : int
        ceiling on buffer size
    episodes_to_store : int
        number of full episodes stored in replay buffer
    num_replay_goals : int
        number of transitions to serve as replay goals during goal replay
    trans_per_attempt : int
        number of the transitions created for each attempt (i.e, action replay
        + goal replay + subgoal testing)
    buffer_size : int
        size of the replay buffer
    batch_size : int
        SGD batch size
    replay_buffer : hac.ExperienceBuffer
        the replay buffer object
    temp_goal_replay_storage : list
        buffer to store not yet finalized goal replay transitions
    actor : hac.Actor
        the actor class
    critic : hac.Critic
        the critic class
    noise_perc : array_like
        percentage of range for each action that will be used to define the
        standard deviation of the noise injected to an action
    maxed_out : bool
        flag to indicate when layer has ran out of attempts to achieve goal.
        This will be important for subgoal testing.
    subgoal_penalty : float
        penalty associated with missing a subgoal
    """

    def __init__(self, layer_number, flags, env, sess, agent_params):
        """Instantiate the Layer object.

        Parameters
        ----------
        layer_number : int
            the level of the layer (0 being the lowest)
        flags : argparse.Namespace
            the parsed arguments from the command line (see options.py)
        env : hac.Environment
            the training environment
        sess : tf.Session
            the tensorflow session
        agent_params : dict
            agent-specific parameters
        """
        self.layer_number = layer_number
        self.flags = flags
        self.sess = sess

        # Set time limit for each layer.
        #
        # If agent uses only 1 layer, time limit is the max number of low-level
        # actions allowed in the episode (i.e, env.max_actions).
        if flags.layers > 1:
            self.time_limit = flags.time_scale
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None

        # Initialize Replay Buffer. Below variables determine size of replay
        # buffer.

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 3

        # Number of the transitions created for each attempt
        # (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = \
                (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = \
                (1 + self.num_replay_goals) * self.time_limit + \
                int(self.time_limit/3)

        # Buffer size = transitions per attempt * # attempts per episode *
        # num of episodes stored
        self.buffer_size = min(
            self.trans_per_attempt *
            self.time_limit**(flags.layers-1 - self.layer_number) *
            self.episodes_to_store,
            self.buffer_size_ceiling)

        self.batch_size = 1024
        self.replay_buffer = \
            ExperienceBuffer(self.buffer_size, self.batch_size)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        # Initialize actor and critic networks
        self.actor = Actor(
            sess, env, self.batch_size, self.layer_number, flags)
        self.critic = Critic(sess, env, self.layer_number, flags)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

        # Create flag to indicate when layer has ran out of attempts to achieve
        # goal. This will be important for subgoal testing.
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]

    def add_noise(self, action, env):
        """Add noise to provided action.

        Parameters
        ----------
        action : hac.Agent
            the agent class
        env : hac.Environment
            the training environment

        Returns
        -------
        array_like
            the action with noise
        """
        # Noise added will be percentage of range
        if self.layer_number == 0:
            ac_space = env.action_space
            bounds = (ac_space.high - ac_space.low) / 2
            offset = (ac_space.high + ac_space.low) / 2
        else:
            bounds = env.subgoal_bounds_symmetric
            offset = env.subgoal_bounds_offset

        assert len(action) == len(bounds), \
            "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), \
            "Noise percentage vector must have same dimension as action"

        for i in range(len(action)):
            # Add noise to action.
            action[i] += np.random.normal(0, self.noise_perc[i] * bounds[i])

            # Ensure the action remains within bounds.
            action[i] = max(min(action[i], bounds[i] + offset[i]),
                            -bounds[i] + offset[i])

        return action

    def get_random_action(self, env):
        """Select random action.

        Parameters
        ----------
        env : hac.Environment
            the training environment

        Returns
        -------
        array_like
            a random action, within the bounds that are specified in the
            environment
        """
        if self.layer_number == 0:
            action = np.zeros(env.action_space.shape[0])
        else:
            action = np.zeros(env.subgoal_dim)

        # Each dimension of random action should take some value in the
        # dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                ac_space = env.action_space
                bounds = (ac_space.high - ac_space.low) / 2
                offset = (ac_space.high + ac_space.low) / 2

                action[i] = np.random.uniform(-bounds[i] + offset[i],
                                              bounds[i] + offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0],
                                              env.subgoal_bounds[i][1])

        return action

    def choose_action(self, agent, env, subgoal_test):
        """Select action using an epsilon-greedy policy.

        Parameters
        ----------
        agent : hac.Agent
            the agent class
        env : hac.Environment
            the training environment
        subgoal_test : bool
            TODO

        Returns
        -------
        array_like
            the action by the agent
        str
            the action type, one of: {"Policy", "Noise Policy", "Random"}
        bool
            specifies whether to perform evaluation on the next subgoal
        """
        # If testing mode or testing subgoals, action is output of actor
        # network without noise
        if self.flags.test or subgoal_test:
            return self.actor.get_action(
                np.reshape(self.current_state,
                           (1, len(self.current_state))),
                np.reshape(self.goal, (1, len(self.goal)))
            )[0], "Policy", subgoal_test

        else:
            if np.random.random_sample() > 0.2:
                # Choose noisy action
                action = self.add_noise(self.actor.get_action(
                    np.reshape(self.current_state,
                               (1, len(self.current_state))),
                    np.reshape(self.goal, (1, len(self.goal))))[0], env)
                action_type = "Noisy Policy"

            # Otherwise, choose random action
            else:
                action = self.get_random_action(env)
                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

            return action, action_type, next_subgoal_test

    def perform_action_replay(self, hindsight_action, next_state, goal_status):
        """Create action replay transition.

        This is done by evaluating hindsight action given original goal.

        Parameters
        ----------
        hindsight_action : array_like
            TODO
        next_state : array_like
            the next observation
        goal_status : list of bool
            whether the goal was achieved by each layer
        """
        # Determine reward (0 if goal achieved, -1 otherwise) and finished
        # boolean. The finished boolean is used for determining the target for
        # Q-value updates
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False

        # Transition will take the form [old state, hindsight_action, reward,
        # next_state, goal, terminate boolean, None]
        transition = [self.current_state, hindsight_action, reward, next_state,
                      self.goal, finished, None]

        # Add action replay transition to layer's replay buffer
        self.replay_buffer.add(deepcopy(transition))

    def create_prelim_goal_replay_trans(self,
                                        hindsight_action,
                                        next_state,
                                        env,
                                        total_layers):
        """Create initial goal replay transitions.

        Create transition evaluating hindsight action for some goal to be
        determined in future. Goal will be ultimately be selected from states
        layer has traversed through.

        Transition will be in the form:

        [
            old state,
            hindsight action,
            reward = None,
            next state,
            goal = None,
            finished = None,
            next state projected to subgoal/end goal space
        ]

        Parameters
        ----------
        hindsight_action : array_like
            TODO
        next_state : array_like
            the next observation
        env : hac.Environment
            the training environment
        total_layers : int
            number of layers in the hierarchical network
        """
        if self.layer_number == total_layers - 1:
            hindsight_goal = env.project_state_to_end_goal(env.sim, next_state)
        else:
            hindsight_goal = env.project_state_to_subgoal(env.sim, next_state)

        transition = [self.current_state, hindsight_action, None, next_state,
                      None, None, hindsight_goal]

        self.temp_goal_replay_storage.append(np.copy(transition))


    def finalize_goal_replay(self, goal_thresholds):
        """Finalize goal replay.

        This is done by filling in goal, reward, and finished boolean for the
        preliminary goal replay transitions created before.

        Parameters
        ----------
        goal_thresholds : array_like
            goal achievement thresholds. If the agent is within the threshold
            for each dimension, the end goal has been achieved and the reward
            of 0 is granted.
        """
        # Choose transitions to serve as goals during goal replay.  The last
        # transition will always be used
        num_trans = len(self.temp_goal_replay_storage)

        num_replay_goals = self.num_replay_goals
        # If fewer transitions that ordinary number of replay goals, lower
        # number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros(num_replay_goals)
        indices[:num_replay_goals-1] = np.random.randint(
            num_trans, size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        # For each selected transition, update the goal dimension of the
        # selected transition and all prior transitions by using the next state
        # of the selected transition as the new goal.  Given new goal, update
        # the reward and finished boolean as well.
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            new_goal = trans_copy[int(indices[i])][6]
            # for index in range(int(indices[i])+1):
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(
                    new_goal, trans_copy[index][6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                # Add finished transition to replay buffer
                # if self.layer_number == 1:
                    # print("\nNew Goal: ", new_goal)
                    # print("Upd Trans %d: " % index, trans_copy[index])

                self.replay_buffer.add(trans_copy[index])

        # Clear storage for preliminary goal replay transitions at end of goal
        # replay
        self.temp_goal_replay_storage = []

    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):
        """Create transition penalizing subgoal if necessary.

        The target Q-value when this transition is used will ignore next state
        as the finished boolena = True. Change the finished boolean to False,
        if you would like the subgoal penalty to depend on the next state.

        Parameters
        ----------
        subgoal : array_like
            TODO
        next_state : array_like
            next step observation
        high_level_goal_achieved : bool
            specifies whether the goal assigned by the layer one level above
            was achieved
        """
        transition = [self.current_state, subgoal, self.subgoal_penalty,
                      next_state, self.goal, True, None]

        self.replay_buffer.add(deepcopy(transition))

    def return_to_higher_level(self,
                               max_lay_achieved,
                               agent,
                               env,
                               attempts_made):
        """Determine whether layer is finished training.

        This method returns to higher level if:

        (i)   a higher level goal has been reached,
        (ii)  maxed out episode time steps (env.max_actions),
        (iii) not testing and layer is out of attempts, and
        (iv)  testing, layer is not the highest level, and layer is out of
              attempts.

        NOTE: during testing, highest level will continue to ouput subgoals
        until either

        (i)   the maximum number of episdoe time steps or
        (ii)  the end goal has been achieved.

        Parameters
        ----------
        max_lay_achieved : int
            highest layer that achieved its goal
        agent : hac.Agent
            the agent class
        env : hac.Environment
            the training environment
        attempts_made : int
            number of actions the layer performed in the current rollout

        Returns
        -------
        bool
            TODO
        """
        # Return to previous level when any higher level goal achieved.
        #
        # NOTE: if not testing and agent achieves end goal, training will
        # continue until out of time (i.e., out of time steps or highest level
        # runs out of attempts).  This will allow agent to experience being
        # around the end goal.
        if max_lay_achieved is not None \
                and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not self.flags.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to
        # achieve goal
        elif self.flags.test \
                and self.layer_number < self.flags.layers - 1 \
                and attempts_made >= self.time_limit:
            return True

        else:
            return False

    def train(self, agent, env, subgoal_test=False, episode_num=None):
        """Perform the training procedure.

        Learn to achieve goals with actions belonging to appropriate time
        scale. "goal_array" contains the goal states for the current layer and
        all higher layers

        Parameters
        ----------
        agent : hac.Agent
            the agent class
        env : hac.Environment
            the training environment
        subgoal_test : bool, optional
            TODO
        episode_num : int, optional
            number of episodes since training has begun. Used for logging
            purposes

        Returns
        -------
        list of bool
            specifies whether the goal was achieved by each layer
        int
            highest layer that achieved its goal
        """
        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This
        # will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is
        # bottom layer
        if self.layer_number == 0 \
                and self.flags.show and self.flags.layers > 1:
            env.display_subgoals(agent.goal_array)

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        while True:
            # Select action to achieve goal state using epsilon-greedy policy
            # or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(
                agent, env, subgoal_test)

            # If next layer is not bottom level, propose subgoal for next layer
            # to achieve and determine whether that subgoal should be tested
            if self.layer_number > 0:
                agent.goal_array[self.layer_number - 1] = action
                goal_status, max_lay_achieved = \
                    agent.layers[self.layer_number - 1].train(
                        agent, env, next_subgoal_test, episode_num)

            # If layer is bottom level, execute low-level action
            else:
                next_state, _, _, _ = env.step(action)

                # Increment steps taken
                agent.steps_taken += 1
                # print("Num Actions Taken: ", agent.steps_taken)

                if agent.steps_taken >= env.max_actions:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was
                # achieved and, if applicable, the highest layer whose goal
                # was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1

            # TODO: tensorboard
            # Print if goal from current layer as been achieved
            if goal_status[self.layer_number]:
                if self.layer_number < self.flags.layers - 1:
                    print("SUBGOAL ACHIEVED")
                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" %
                      (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == self.flags.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(
                        env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(
                        env.sim, agent.current_state))

            # Perform hindsight learning using action actually executed
            # (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as
                # hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_subgoal(
                        env.sim, agent.current_state)

            # Next, create hindsight transitions if not testing
            if not self.flags.test:
                # Create action replay transition by evaluating hindsight
                # action given current goal
                self.perform_action_replay(
                    hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and
                # reward in these transitions will be finalized when this layer
                # has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(
                    hindsight_action, agent.current_state, env,
                    self.flags.layers)

                # Penalize subgoals if subgoal testing and subgoal was missed
                # by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test \
                        and agent.layers[self.layer_number - 1].maxed_out:
                    self.penalize_subgoal(
                        action, agent.current_state,
                        goal_status[self.layer_number])

            # Print summary of transition
            if self.flags.verbose:
                print("\nEpisode %d, Training Layer %d, Attempt %d" % (
                    episode_num, self.layer_number, attempts_made))
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == self.flags.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(
                        env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(
                        env.sim, agent.current_state))
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)

            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            if (max_lay_achieved is not None
                and max_lay_achieved >= self.layer_number) \
                    or agent.steps_taken >= env.max_actions \
                    or attempts_made >= self.time_limit:

                if self.layer_number == self.flags.layers-1:
                    print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set
                # maxed out flag to true
                if attempts_made >= self.time_limit \
                        and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)

                # If not testing, finish goal replay by filling in missing goal
                # and reward values before returning to prior level.
                if not self.flags.test:
                    if self.layer_number == self.flags.layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek
                # a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env,
                                               attempts_made):
                    return goal_status, max_lay_achieved

    def learn(self, num_updates):
        """Update the trainable variables of the actor and critic networks.

        Parameters
        ----------
        num_updates : int
            number of times to compute and apply gradient updates before
            exiting
        """
        for _ in range(num_updates):
            # Update weights of non-target networks
            if self.replay_buffer.size >= self.batch_size:
                old_states, actions, rewards, new_states, goals, is_terminals \
                    = self.replay_buffer.get_batch()

                self.critic.update(
                    old_states, actions, rewards, new_states, goals,
                    self.actor.get_action(new_states, goals), is_terminals)
                action_derivs = self.critic.get_gradients(
                    old_states, goals,
                    self.actor.get_action(old_states, goals))
                self.actor.update(old_states, goals, action_derivs)

        # =================================================================== #
        # To use target networks comment for loop above and uncomment for     #
        # loop below.                                                         #
        # =================================================================== #

        # for _ in range(num_updates):
        #     # Update weights of non-target networks
        #     if self.replay_buffer.size >= self.batch_size:
        #         old_states, actions, rewards, new_states, goals, \
        #             is_terminals = self.replay_buffer.get_batch()
        #
        #         self.critic.update(
        #             old_states, actions, rewards, new_states, goals,
        #             self.actor.get_target_action(new_states,goals),
        #             is_terminals)
        #         action_derivs = self.critic.get_gradients(
        #             old_states, goals,
        #             self.actor.get_action(old_states, goals))
        #         self.actor.update(old_states, goals, action_derivs)
        #
        # # Update weights of target networks
        # self.sess.run(self.critic.update_target_weights)
        # self.sess.run(self.actor.update_target_weights)
