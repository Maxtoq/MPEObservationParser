import random
import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Action, Entity

LANDMARK_SIZE = 0.1
OBJECT_SIZE = 0.15
OBJECT_MASS = 1.0
AGENT_SIZE = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

def obj_callback(agent, world):
    action = Action()
    action.u = np.zeros((world.dim_p))
    action.c = np.zeros((world.dim_c))
    return action

class Color_Entity(Entity):
    def __init__(self):
        super(Color_Entity, self).__init__()

        self.num_color = 0

        # Get the color based on the number
    def num_to_color(self, color):
        match color:
            #Black
            case 1:
                color = [0.3, 0.3, 0.3]
            # Red
            case 2:
                color = [1, 0.22745, 0.18431]
            # Blue
            case 3:
                color = [0, 0.38, 1]
            # Green
            case 4:
                color = [0.2, 0.78 , 0.35]
            # Yellow
            case 5:
                color = [1, 0.8 , 0]
            # Purple
            case 6:
                color = [0.8, 0.21, 0.98]

        return color

    # Get the color based on the number
    def color_to_num(self, color):
        match color:
            #Black
            case [0.3, 0.3, 0.3]:
                color = 1
            # Red
            case [1, 0.22745, 0.18431]:
                color = 2
            # Blue
            case [0, 0.38, 1]:
                color = 3
            # Green
            case [0.2, 0.78 , 0.35]:
                color = 4
            # Yellow
            case [1, 0.8 , 0]:
                color = 5
            # Purple
            case [0.8, 0.21, 0.98]:
                color = 6

        return color

# properties of landmark entities
class Landmark(Color_Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of object entities
class Object(Color_Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):
    def __init__(self, nb_agents, nb_objects):
        super(PushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        # Corresponding landmarks
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]
        # Distances between objects and their landmark
        self.obj_lm_dists = np.zeros(self.nb_objects)
        # Shaping reward based on distances between objects and lms
        self.shaping_reward = 0.0
        # Control inertia
        self.damping = 0.8

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.2, max_dist=1.5):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        #color = self.pick_color()
        # Object
        self.objects[obj_i].name = 'object %d' % len(self.objects)
        self.objects[obj_i].color = color
        self.objects[obj_i].size = OBJECT_SIZE
        self.objects[obj_i].initial_mass = OBJECT_MASS
        # Landmark
        self.landmarks[obj_i].name = 'landmark %d' % len(self.landmarks)
        self.landmarks[obj_i].collide = False
        self.landmarks[obj_i].color = color
        self.landmarks[obj_i].size = LANDMARK_SIZE
        # Set initial positions
        # # Fixed initial pos
        # self.objects[obj_i].state.p_pos = np.zeros(2)
        # self.landmarks[obj_i].state.p_pos = np.array([-0.5, -0.5])
        # return
        if min_dist is not None:
            while True:
                self.objects[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                self.landmarks[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                dist = get_dist(self.objects[obj_i].state.p_pos, 
                                self.landmarks[obj_i].state.p_pos)
                if dist > min_dist and dist < max_dist:
                    break
        else:
            dist = get_dist(self.objects[obj_i].state.p_pos, 
                            self.landmarks[obj_i].state.p_pos)
        # Set distances between objects and their landmark
        self.obj_lm_dists[obj_i] = dist

    def step(self):
        # s
        last_obj_lm_dists = np.copy(self.obj_lm_dists)
        super().step()
        # s'
        # Compute shaping reward
        self.shaping_reward = 0.0
        for obj_i in range(self.nb_objects):
            # Update dists
            self.obj_lm_dists[obj_i] = get_dist(
                self.objects[obj_i].state.p_pos,
                self.landmarks[obj_i].state.p_pos)
            # Compute reward
            self.shaping_reward += last_obj_lm_dists[obj_i] \
                                    - self.obj_lm_dists[obj_i]

    # Integrate state with walls blocking entities on each side
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) \
                        + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            # Check for wall collision
            temp_pos = entity.state.p_pos + entity.state.p_vel * self.dt
            # West wall
            if temp_pos[0] - entity.size < -1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = -1.0 + entity.size
            # East wall
            if temp_pos[0] + entity.size > 1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = 1.0 - entity.size
            # North wall
            if temp_pos[1] - entity.size < -1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = -1.0 + entity.size
            # South wall
            if temp_pos[1] + entity.size > 1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = 1.0 - entity.size
            entity.state.p_pos += entity.state.p_vel * self.dt
                
        

class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4, 
                   collision_pen=1, relative_coord=True, dist_reward=False, 
                   reward_done=50, step_penalty=0.1, obj_lm_dist_range=[0.2, 1.5]):
        world = PushWorld(nb_agents, nb_objects)
        # add agent
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[i % 3] = 1.0
        # Objects and landmarks
        self.nb_objects = nb_objects
        
        # Set list of colors
        colors = []
        color = [1,1,1]
        for i, object in enumerate(world.objects):
            # color = np.random.uniform(0, 1, world.dim_color)
            # Pick a color that is not already taken
            while True:
                same = False
                # Pick a color number
                color = np.random.randint(1,7)
        
                for c in colors:
                    if c == color:
                        same = True

                if same == False:
                    break
            colors.append(color)

            object.name = 'object %d' % i
            object.num_color = color
            object.color = object.num_to_color(color)
            object.size = OBJECT_SIZE
            object.initial_mass = OBJECT_MASS

        for land in world.landmarks:
            land.collide = False
            land.size = LANDMARK_SIZE

            # Take a random color
            color = random.choice(colors)
            colors.remove(color)

            # Corresponding Landmarks
            for i, object in enumerate(world.objects):
                if object.num_color == color:
                    land.name = 'landmark %d' % i
                    land.num_color = color
                    land.color = land.num_to_color(color)
        
        self.obj_lm_dist_range = obj_lm_dist_range
        # Scenario attributes
        self.obs_range = obs_range
        self.relative_coord = relative_coord
        self.dist_reward = dist_reward
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # Reward for completing the task
        self.reward_done = reward_done
        # Penalty for step in the environment
        self.step_penalty = step_penalty
        # make initial conditions
        self.reset_world(world)
        return world


    def done(self, agent, world):
        # Done if all objects are on their landmarks
        return self._done_flag

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)
        # Check if init positions are valid
        if init_pos is not None:
            if (len(init_pos["agents"]) != self.nb_agents or 
                len(init_pos["objects"]) != self.nb_objects or
                len(init_pos["landmarks"]) != self.nb_objects):
                print("ERROR: The initial positions {} are not valid.".format(
                    init_pos))
                exit(1)
        # world.reset()
        # Agents' initial pos
        # # Fixed initial pos
        # world.agents[0].state.p_pos = np.array([0.5, -0.5])
        # world.agents[1].state.p_pos = np.array([-0.5, 0.5])
        for i, agent in enumerate(world.agents):
            if init_pos is None:
                agent.state.p_pos = np.random.uniform(
                    -1 + agent.size, 1 - agent.size, world.dim_p)
            else:
                agent.state.p_pos = np.array(init_pos["agents"][i])
            agent.state.c = np.zeros(world.dim_c)
        # Objects and landmarks' initial pos
        for i, object in enumerate(world.objects):
            if init_pos is None:
                while True:
                    object.state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    world.landmarks[i].state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    dist = get_dist(object.state.p_pos, 
                                    world.landmarks[i].state.p_pos)
                    if (self.obj_lm_dist_range is None  or 
                        (dist > self.obj_lm_dist_range[0] and 
                         dist < self.obj_lm_dist_range[1])):
                        break
            else:
                object.state.p_pos = np.array(init_pos["objects"][i])
                world.landmarks[i].state.p_pos = np.array(init_pos["landmarks"][i])
                dist = get_dist(object.state.p_pos, 
                                world.landmarks[i].state.p_pos)
            # Set distances between objects and their landmark
            world.obj_lm_dists[i] = dist
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        self._done_flag = False

    def reward(self, agent, world):
        # Reward = -1 x squared distance between objects and corresponding landmarks
        dists = [get_dist(obj.state.p_pos, 
                          world.landmarks[i].state.p_pos)
                    for i, obj in enumerate(world.objects)]
        print("Dist: ")
        print(dists)
        dists = []
        for obj in world.objects:
            print(obj.num_color)
            for land in world.landmarks:
                print("+" + str(land.num_color))
                if obj.num_color == land.num_color:
                    dists.append(get_dist(obj.state.p_pos, 
                          land.state.p_pos))
                    break
        print(dists)
        # rew = -sum([pow(d * 3, 2) for d in dists])
        # rew = -sum(dists)
        # rew = -sum(np.exp(dists))
        # Shaped reward
        shaped = 100 * world.shaping_reward
        # if world.shaping_reward > 0:
        #     shaped = 100 * world.shaping_reward
        # else:
        #     shaped = 10 * world.shaping_reward
            # shaped = 0
        rew = -self.step_penalty + shaped

        # Reward if task complete
        self._done_flag = all(d <= LANDMARK_SIZE for d in dists)
        if self._done_flag:
            rew += self.reward_done

        # Penalty for collision between agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    # print("COLLISION")
                    rew -= self.collision_pen
        # Penalty for collision with wall
        # if (agent.state.p_pos - agent.size <= -1).any() or \
        #    (agent.state.p_pos + agent.size >= 1).any():
        #    rew -= self.collision_pen
        return rew

    def observation(self, agent, world):
        # Observation:
        #  - Agent state: position, velocity
        #  - Other agents and objects:
        #     - If in sight: [relative x, relative y, v_x, v_y]
        #     - If not: [0, 0, 0, 0]
        #  - Landmarks:
        #     - If in sight: [relative x, relative y]
        #     - If not: [0, 0]
        # => Full observation dim = 2 + 2 + 5 x (nb_agents_objects - 1) + 3 x (nb_landmarks)
        # All distances are divided by max_distance to be in [0, 1]
        entity_obs = []
        for entity in world.agents:
            if entity is agent: continue
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range, entity.state.p_vel
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        [1.0], # Bit saying entity is observed
                        (entity.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                        entity.state.p_vel # Velocity
                        # (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                    )))
                # Pos: absolute
                else:
                    entity_obs.append(np.concatenate((
                        # [1.0], entity.state.p_pos, entity.state.p_vel
                        entity.state.p_pos, entity.state.p_vel
                    )))
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
                else:
                    entity_obs.append(np.zeros(4))
        for entity in world.objects:
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range, entity.state.p_vel
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        [1.0], # Bit saying entity is observed
                        (entity.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                        entity.state.p_vel, # Velocity
                        [entity.num_color]
                        # (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                    )))
                # Pos: absolute
                else:
                    entity_obs.append(np.concatenate((
                        # [1.0], entity.state.p_pos, entity.state.p_vel
                        entity.state.p_pos, entity.state.p_vel, [entity.num_color]
                    )))
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0]))
                else:
                    entity_obs.append(np.zeros(5))
        for entity in world.landmarks:
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        [1.0], 
                        (entity.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                        [entity.num_color]
                    )))
                    # entity_obs.append(
                    #     entity.state.p_pos - agent.state.p_pos
                    # )
                # Pos: absolute
                else:
                    # entity_obs.append(np.concatenate((
                    #     [1.0], entity.state.p_pos
                    # )))
                    entity_obs.extend(entity.state.p_pos, entity.num_color)
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0]))
                else:
                    entity_obs.append(np.zeros(3))

        # Communication


        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_obs)