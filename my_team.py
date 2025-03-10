import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
    """
    A reflex agent that seeks food, avoids enemies when in enemy territory, 
    and chases enemies when in its own field.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_carrying = 0
        self.last_position = None
        self.turns_without_move = 0
        self.max_turns_without_move = 1  # Set the number of turns after which we force a move

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        self.food_carrying = game_state.get_agent_state(self.index).num_carrying

        border_positions = self.get_border_positions(game_state)
        min_border_distance = min([self.get_maze_distance(my_pos, b) for b in border_positions])

        best_action = None
        min_enemy_distance = float('inf')

        # Check if Pacman has moved
        if self.last_position == my_pos:
            self.turns_without_move += 1
        else:
            self.turns_without_move = 0  # Reset counter if position changed

        # If Pacman hasn't moved for too long, force a move
        if self.turns_without_move >= self.max_turns_without_move:
            print("Pacman hasn't moved for a while, forcing a move!")
            self.turns_without_move = 0  # Reset counter after forcing move
            return self.force_move(game_state, actions)

        # Update the last position
        self.last_position = my_pos

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            enemy_distances = [self.get_maze_distance(new_pos, g.get_position()) for g in ghosts]
            try:
                closest_enemy_dist = min(enemy_distances)
            except ValueError:
                closest_enemy_dist = 999

            # *1. If carrying food and near border, prioritize returning*
            if self.food_carrying >= 3 and min_border_distance <= 10 or closest_enemy_dist < 3:
                return self.return_to_border(game_state, actions, border_positions)

            # *2. Chase intruders in own field*
            if self.is_in_own_territory(my_pos) == False and ghosts:
                return self.chase_enemy(game_state, actions, ghosts)

            # *3. Escape enemies when in enemy territory*
            if self.is_in_enemy_territory(my_pos) == False and ghosts:
                return self.smart_escape(game_state, actions, ghosts)

            # *4. Default food-seeking behavior, ensuring safety*
            return self.safe_seek_food(game_state, actions, ghosts)

    def force_move(self, game_state, actions):
        # Implement a move that will force Pacman to take an action
        # This can be any move, for instance, returning to a default position
        # or just making a random move if you want to ensure movement.
        return actions[0]  # Just return the first legal action (or implement a better strategy)


    def return_to_border(self, game_state, actions, border_positions):
        """Safely return to our own side when carrying food."""
        best_action = None
        best_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = min([self.get_maze_distance(new_pos, b) for b in border_positions])
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action if best_action else random.choice(actions)

    def chase_enemy(self, game_state, actions, ghosts):
        """Chase enemy invaders when in our own field."""
        best_action = None
        min_enemy_distance = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            enemy_distances = [self.get_maze_distance(new_pos, g.get_position()) for g in ghosts]
            closest_enemy_dist = min(enemy_distances)

            if closest_enemy_dist < min_enemy_distance:
                best_action = action
                min_enemy_distance = closest_enemy_dist

        return best_action if best_action else random.choice(actions)

    def smart_escape(self, game_state, actions, ghosts):
        """Improved escape strategy in enemy territory."""
        enemy_positions = [g.get_position() for g in ghosts]
        safe_actions = []

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            min_dist = min([self.get_maze_distance(new_pos, g) for g in enemy_positions])
            if min_dist >= 3:
                safe_actions.append((action, min_dist))

        if safe_actions:
            return max(safe_actions, key=lambda x: x[1])[0]  # Pick the safest option

        return random.choice(actions)  # If no safe actions, pick randomly

    def safe_seek_food(self, game_state, actions, ghosts):
        """Move towards the nearest food while avoiding enemies."""
        safe_actions = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            if all(self.get_maze_distance(new_pos, g.get_position()) >= 3 for g in ghosts):
                safe_actions.append(action)
        
        if safe_actions:
            actions = safe_actions  # Prefer actions that keep distance from enemies
        
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

    def is_in_enemy_territory(self, position):
        return position[0] > (self.start[0] // 2)

    def is_in_own_territory(self, position):
        return position[0] <= (self.start[0] // 2)

    def get_border_positions(self, game_state):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        border_x = (self.start[0] // 2) if self.red else (self.start[0] // 2) + 1
        return [(border_x, y) for y in range(height) if not game_state.has_wall(border_x, y)]


class DefensiveAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free but also collects food and behaves like an attacker when no enemies are present.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        remaining_food = len(self.get_food(game_state).as_list())
        
        # If food is scarce, prioritize defense
        if remaining_food < 8 or len(invaders) > 0:
            values = [self.evaluate(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return random.choice(best_actions)
        else:
            actions = game_state.get_legal_actions(self.index)
        
            # Identify enemies
            my_pos = game_state.get_agent_state(self.index).get_position()
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

            # Track food carried
            self.food_carrying = game_state.get_agent_state(self.index).num_carrying
            border_positions = self.get_border_positions(game_state)
            min_border_distance = min([self.get_maze_distance(my_pos, b) for b in border_positions])

            # If carrying a lot of food and near the border, prioritize returning
            if self.food_carrying >= 3 and min_border_distance <= 10:
                best_action = None
                best_dist = float('inf')
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    dist = min([self.get_maze_distance(new_pos, b) for b in border_positions])
                    if dist < best_dist:
                        best_action = action
                        best_dist = dist
                return best_action if best_action else random.choice(actions)
            
            # If enemies are close, escape by maximizing distance and avoiding dead ends
            if ghosts:
                enemy_positions = [g.get_position() for g in ghosts]
                min_ghost_distance = min([self.get_maze_distance(my_pos, g) for g in enemy_positions])
                if min_ghost_distance < 8:
                    best_action = None
                    max_distance = -1
                    escape_actions = []
                    
                    for action in actions:
                        successor = self.get_successor(game_state, action)
                        new_pos = successor.get_agent_state(self.index).get_position()
                        min_dist = min([self.get_maze_distance(new_pos, g) for g in enemy_positions])
                        
                        # Prefer actions that move towards our own field
                        if self.is_in_enemy_territory(new_pos):
                            escape_actions.append((action, min_dist))
                        
                        if min_dist > max_distance:
                            best_action = action
                            max_distance = min_dist
                    
                    # If possible, escape towards our side of the field
                    if escape_actions:
                        best_action = max(escape_actions, key=lambda x: x[1])[0]
                    
                    return best_action if best_action else random.choice(actions)  # Always keep moving

            # Default behavior
            values = [self.evaluate(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]

            food_left = len(self.get_food(game_state).as_list())
            if food_left <= 2:
                best_dist = 9999
                best_action = None
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    pos2 = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(self.start, pos2)
                    if dist < best_dist:
                        best_action = action
                        best_dist = dist
                return best_action

            return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    def get_offensive_action(self, game_state, actions):
        """When no enemies are present and enough food remains, collect food like an attacking agent and escape if carrying food."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_carrying = game_state.get_agent_state(self.index).num_carrying
        border_positions = self.get_border_positions(game_state)
        min_border_distance = min([self.get_maze_distance(my_pos, b) for b in border_positions])

        # Detect enemies within 20 steps
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        nearby_enemies = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        close_enemy = None
        min_enemy_distance = float('inf')
        for enemy in nearby_enemies:
            enemy_pos = enemy.get_position()
            dist = self.get_maze_distance(my_pos, enemy_pos)
            if dist <= 20 and dist < min_enemy_distance:
                close_enemy = enemy
                min_enemy_distance = dist

        # If close enemy is detected, chase it
        if close_enemy:
            best_action = None
            best_dist = float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(new_pos, close_enemy.get_position())
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action if best_action else random.choice(actions)

        # If carrying food and close to border, escape
        if food_carrying >= 2 and min_border_distance <= 15:
            best_action = None
            best_dist = float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                dist = min([self.get_maze_distance(new_pos, b) for b in border_positions])
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action if best_action else random.choice(actions)

        # Otherwise, collect food
        values = [self.evaluate_offense(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def evaluate_offense(self, game_state, action):
        """Evaluates actions based on food collection when no enemies are present."""
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features * {'successor_score': 100, 'distance_to_food': -1}

    def get_border_positions(self, game_state):
        """Finds the border positions where the agent can return to its own side."""
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        border_x = (self.start[0] // 2) if self.red else (self.start[0] // 2) + 1
        return [(border_x, y) for y in range(height) if not game_state.has_wall(border_x, y)]

    def is_in_enemy_territory(self, position):
        """Checks if the given position is in the enemy's side of the field."""
        return position[0] > (self.start[0] // 2)
