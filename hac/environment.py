import numpy as np
import gym
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim, MjViewer


class Environment(gym.Env):
    """Base environment class.

    TODO

    Attributes
    ----------
    name : str TODO remove and just call the separate environments
        name of the environment; adopted from the name of the model
    model : TODO
        TODO
    sim : mujoco_py.MjSim
        TODO
    state_dim : int
        low-level environment observation dimension
    action_dim : int
        low-level action dimension
    action_bounds : array_like
        low-level action bounds
    action_offset : array_like
        TODO
    end_goal_dim : int
        TODO
    subgoal_dim : int
        TODO
    subgoal_bounds : array_like
        range for each dimension of subgoal space
    project_state_to_end_goal : TODO
        TODO
    project_state_to_subgoal : TODO
        TODO
    subgoal_bounds_symmetric : array_like
        TODO
    subgoal_bounds_offset : array_like
        TODO
    end_goal_thresholds : array_like
        goal achievement thresholds. If the agent is within the threshold for
        each dimension, the end goal has been achieved and the reward of 0 is
        granted.
    subgoal_thresholds : array_like
        subgoal achievement thresholds
    initial_state_space : array_like
        initial values for all elements in the state space
    goal_space_train : list of (float, float)
        upper and lower bounds of each element of the goal space during
        training
    goal_space_test : list of (float, float)
        upper and lower bounds of each element of the goal space during
        evaluation
    subgoal_colors : list of str
        colors that are assigned to the subgoal points during visualization
    max_actions : int
        maximum number of atomic actions. This will typically be
        flags.time_scale**(flags.layers).
    visualize : bool
        specifies whether to render the environment
    viewer : mujoco_py.MjViewer
        TODO
    num_frames_skip : int
        number of time steps per atomic action
    """

    def __init__(self,
                 model_name,
                 goal_space_train,
                 goal_space_test,
                 project_state_to_end_goal,
                 end_goal_thresholds,
                 initial_state_space,
                 subgoal_bounds,
                 project_state_to_subgoal,
                 subgoal_thresholds,
                 max_actions=1200,
                 num_frames_skip=10,
                 show=False):
        """Instantiate the Environment object.

        Parameters
        ----------
        model_name : str
            name of the xml file in './mujoco_files/' that the model is
            generated from
        goal_space_train : list of (float, float)
            upper and lower bounds of each element of the goal space during
            training
        goal_space_test : list of (float, float)
            upper and lower bounds of each element of the goal space during
            evaluation
        project_state_to_end_goal : TODO
            TODO
        end_goal_thresholds : array_like
            goal achievement thresholds. If the agent is within the threshold
            for each dimension, the end goal has been achieved and the reward
            of 0 is granted.
        initial_state_space : array_like
            initial values for all elements in the state space
        subgoal_bounds : array_like
            range for each dimension of subgoal space
        project_state_to_subgoal : TODO
            TODO
        subgoal_thresholds : array_like
            subgoal achievement thresholds
        max_actions : int, optional
            maximum number of atomic actions. Defaults to 1200.
        num_frames_skip : int, optional
            number of time steps per atomic action. Defaults to 10.
        show : bool, optional
            specifies whether to render the environment. Defaults to False.
        """
        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("./mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # TODO: abstract the observation and action spaces
        # Set dimensions and ranges of states, actions, and goals in order to
        # configure actor/critic networks
        if model_name == "pendulum.xml":
            # TODO: fill in
            self.observation_space = Box(
                low=0, high=1,
                shape=(2 * len(self.sim.data.qpos) + len(self.sim.data.qvel),)
            )
            # TODO: remove
            self.state_dim = \
                2 * len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
            # State will include (i) joint angles and (ii) joint velocities
            # TODO: fill in
            self.observation_space = Box(
                low=0, high=1,
                shape=(len(self.sim.data.qpos) + len(self.sim.data.qvel),)
            )
            # TODO: remove
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)
        # TODO: fill in
        self.action_space = Box(
            low=0, high=1,
            shape=(len(self.sim.model.actuator_ctrlrange),)
        )
        # TODO: remove
        # low-level action dim
        self.action_dim = len(self.sim.model.actuator_ctrlrange)
        # TODO: remove
        # low-level action bounds
        self.action_bounds = self.sim.model.actuator_ctrlrange[:, 1]
        # Assumes symmetric low-level action ranges
        self.action_offset = np.zeros((len(self.action_bounds)))
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to
        # properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = \
                (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = \
                self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set initial state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = [
            "Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon",
            "Gray", "White", "Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = None
        self.num_frames_skip = num_frames_skip

    def get_state(self):  # TODO: abstract the method
        """Get state, which concatenates joint positions and velocities."""
        if self.name == "pendulum.xml":
            return np.concatenate(
                [np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos),
                 self.sim.data.qvel]
            )
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def reset(self):
        """Reset simulation to state within initial state specified by user.

        Returns
        -------
        array_like
            the initial observation
        """
        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(
                self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(
                self.initial_state_space[len(self.sim.data.qpos) + i][0],
                self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        # Return state
        return self.get_state()

    def step(self, action):
        """Advance the simulation by one step.

        This method executes the low-level action. This is done for number of
        frames specified by num_frames_skip.

        Parameters
        ----------
        action : array_like
            the low level primitive action

        Returns
        -------
        array_like
            the next observation
        float
            reward
        bool
            done mask
        dict
            extra info (set to an empty dictionary by default)
        """
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state(), None, None, {}  # FIXME

    def display_end_goal(self, end_goal):  # TODO: abstract the method
        """Visualize end goal.

        This function may need to be adjusted for new environments.

        Parameters
        ----------
        end_goal : array_like
            the desired end goals to be displayed

        Raises
        ------
        AssertionError
            If the goals to be displayed for the given model are not explicitly
            defined in the method.
        """
        # Goal can be visualized by changing the location of the relevant site
        # object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array(
                [0.5*np.sin(end_goal[0]), 0, 0.5*np.cos(end_goal[0])+0.6])
        elif self.name == "ur5.xml":
            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            t_1_0 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0.089159],
                              [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                              [np.sin(theta_1), np.cos(theta_1), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                              [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                              [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                              [0, 1, 0, 0],
                              [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                              [0, 0, 0, 1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = t_1_0.dot(t_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]
        else:
            assert False, \
                "Provide display end goal function in environment.py file"

    def get_next_goal(self, test):  # TODO: abstract the method
        """Return an end goal.

        Parameters
        ----------
        test : bool
            False if training, True if performing evaluations

        Returns
        -------
        array_like
            TODO
        """
        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":
            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(shape=(self.end_goal_dim,))
                end_goal[0] = np.random.uniform(self.goal_space_test[0][0],
                                                self.goal_space_test[0][1])
                end_goal[1] = np.random.uniform(self.goal_space_test[1][0],
                                                self.goal_space_test[1][1])
                end_goal[2] = np.random.uniform(self.goal_space_test[2][0],
                                                self.goal_space_test[2][1])

                # Next need to ensure chosen joint angles result in achievable
                # task (i.e., desired end effector position is above ground)
                theta_1 = end_goal[0]
                theta_2 = end_goal[1]
                theta_3 = end_goal[2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                # upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                t_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 1, 0.089159], [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference
                # frame
                t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                                  [np.sin(theta_1), np.cos(theta_1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference
                # frame
                t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                                  [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                                  [0, 0, 0, 1]])

                forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(
                    forearm_pos_3)[:3]
                wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                    wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be
                # reached
                if np.absolute(end_goal[0]) > np.pi/4 \
                        and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True

        elif not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],
                                                self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, \
                "Need goal space for testing. Set goal_space_test variable " \
                "in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(
                    self.goal_space_test[i][0], self.goal_space_test[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal

    def display_subgoals(self, subgoals):  # TODO: abstract the method
        """Visualize all subgoals.

        Parameters
        ----------
        subgoals : array_like
            the subgoals to be displayed (e.g. desired positions)
        """
        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array(
                    [0.5*np.sin(subgoals[subgoal_ind][0]), 0,
                     0.5*np.cos(subgoals[subgoal_ind][0])+0.6])

                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
            elif self.name == "ur5.xml":
                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                t_1_0 = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.089159],
                                  [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference
                # frame
                t_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],
                                  [np.sin(theta_1), np.cos(theta_1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference
                # frame
                t_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                                  [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                                  [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                t_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                                  [0, 0, 0, 1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = t_1_0.dot(t_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(
                    forearm_pos_3)[:3]
                wrist_1_pos = t_1_0.dot(t_2_1).dot(t_3_2).dot(t_4_3).dot(
                    wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3*(i-1) + j] = \
                        np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

                subgoal_ind += 1
            else:
                # Visualize desired gripper position, which is elements 18-21
                # in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
