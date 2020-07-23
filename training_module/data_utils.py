import copy
import json
try:
    import zstd
except:
    pass
import numpy as np


"""
**** GLOBAL VARIABLES ****
These are set to ensure that replay downloads, training, and live bots are all
based on the same assumptions, e.g., the top-ranked player is so-and-so.
"""

MAP_SIZES = [32, 40, 48, 56, 64]

class Game(object):
    """
    Can be used iteratively in a live game, or used to parse replays.
    
    'Load' methods are used to populate values from a relay.
    'Parse' methods are used on individual frames (used for both replays and iteratively)
    """
    def __init__(self):
        self.production = []
        self.moves = []
        self.generate = []
        self.entities = []
        self.energy = []
        self.deposited = []
        self.meta = None
        self.map_shape = None
        self.factories = []
        self.dropoffs = []
        self.events = []
        
        self.replay = None # Store the whole replay for easier debugging
    
    def get_replay(self):
        return self.replay
    
    def parse_events(self, frame):
        frame_events = []
        for e in frame['events']:
            if e['type'] == 'spawn':
                event = [0, e['location']['x'], e['location']['y'], e['id'], e['owner_id']]
            elif e['type'] == 'construct':
                event = [1, e['location']['x'], e['location']['y'], e['id'], e['owner_id']]
            elif e['type'] == 'shipwreck':
                event = [2, e['location']['x'], e['location']['y']] + [x for x in e['ships']]
            else:
                assert False, print(e)
            
            frame_events.append(event)
        return frame_events
    
    def parse_factories(self):
        players = self.meta_data['players']
        factories = []
        assert list(range(len(players))) == [x['player_id'] for x in players], print(players)
        for i in range(len(players)):
            f = players[i]['factory_location']
            factories.append([f['x'], f['y']])
        
        return np.array(factories)
    
    def load_dropoffs(self):
        dropoffs = []
        for f in self.events:
            prev = [] if len(dropoffs) == 0 else dropoffs[-1]
            frame_dropoffs = [] + prev # Dropoffs seem to be permanent
            for e in f:
                if e[0] == 1:
                    frame_dropoffs.append([e[1], e[2], e[4]])
            dropoffs.append(frame_dropoffs)
        return dropoffs
    
    
    def load_replay(self, path: str, meta_only=False):
        with open(path, 'rb') as infile:
            raw_replay = zstd.loads(infile.read()).decode()
        replay = json.loads(raw_replay)
        
        self.path = path
        
        self.replay = replay
    
        meta_keys = ['ENGINE_VERSION', 'GAME_CONSTANTS', 'REPLAY_FILE_VERSION',
                     'game_statistics', 'map_generator_seed',
                     'number_of_players', 'players']
                     
        self.meta_data = {key: replay[key] for key in meta_keys}
        
        #self.meta_data['GAME_CONSTANTS']['DEFAULT_MAP_HEIGHT'] = 64
        
        if meta_only:
            return
        
        self.events = [self.parse_events(f) for f in replay['full_frames']]
        self.factories = self.parse_factories()
        
        self.production = self.load_production(replay)
        self.moves, self.generate = self.load_moves(replay)
        self.entities, self.ship_ids = self.load_entities(replay)
        self.energy = self.load_energy(replay)
        self.deposited = self.load_deposited(replay)
        self.dropoffs = self.load_dropoffs()
        
        # player_names = [x['name'].split(' ')[0] for x in meta_data['players']]
        
        # Some of these need to be trimmed
        
        # First is just an init frame
        self.events = self.events[1:]
        
        # Last reflects what the production will be if moves were made on last frame
        # (but there aren't any on last). Also, we don't care what the production looks like
        # on the last frame (because we will have already made our last move).
        # The indexing is weird because the replays show what production would be after moves
        # are made.
        self.production = self.production[:-2]
        self.dropoffs = self.dropoffs[:-2]
        
        # As if moved after last frame, but there are no moves
        self.energy = self.energy[:-1]
        self.deposited = self.deposited[:-1]

    def load_production(self, replay):
        pm = replay['production_map']
        assert list(pm['grid'][0][0].keys()) == ['energy']
        raw_energy_grid = pm['grid']
        energy_grid = []
        for row in raw_energy_grid:
            energy_grid.append([x['energy'] for x in row])
        first_frame = np.array(energy_grid)

        production = []
        for frame in replay['full_frames']:
            prev_frame = first_frame if len(production) == 0 else production[-1]
            current_frame = prev_frame.copy()
            for c in frame['cells']:
                current_frame[c['y'], c['x']] = c['production']
            production.append(current_frame)
        return np.array(production)

    def load_moves(self, replay):
        map_size = self.meta_data['GAME_CONSTANTS']['DEFAULT_MAP_HEIGHT'] # Assuming square
        num_players = int(self.meta_data['number_of_players'])

        valid_moves = ['o', 'n', 'e', 's', 'w', 'c']
        moves = []
        generate = np.zeros((len(replay['full_frames']) - 2, num_players), dtype=np.uint8)
        for ix, frame in enumerate(replay['full_frames'][1:-1]): # No moves on first or last frames
            frame_moves = np.zeros((map_size, map_size, num_players), dtype=np.uint8)
            for pid in range(num_players):
                if str(pid) not in frame['moves']:
                    continue
                for move in frame['moves'][str(pid)]:

                    if move['type'] == 'm':
                        mid = move['id']
                        ent = frame['entities'][str(pid)][str(mid)]
                        assert move['direction'] in valid_moves
                        frame_moves[ent['y'], ent['x'], pid] = valid_moves.index(move['direction'])
                    elif move['type'] == 'g':
                        generate[ix, pid] = 1
                    elif move['type'] == 'c':
                        mid = move['id']
                        ent = frame['entities'][str(pid)][str(mid)]
                        frame_moves[ent['y'], ent['x'], pid] = valid_moves.index('c')
                    else:
                        assert False, print(move)
            moves.append(frame_moves)

        return np.array(moves), generate


    def load_entities(self, replay):
        
        map_size = self.meta_data['GAME_CONSTANTS']['DEFAULT_MAP_HEIGHT'] # Assuming square
        num_players = int(self.meta_data['number_of_players'])
        num_features = 2
        entities = []
        ship_ids = []
        #ship_player_ids = []
        for ix, frame in enumerate(replay['full_frames'][1:-1]): # No entities on first frame (last doesn't matter)
            frame_entities = np.zeros((map_size, map_size, num_features+num_players), dtype=np.int32)
            frame_ship_ids = np.zeros((map_size, map_size), dtype=np.float32)
            for pid in range(num_players):
                for ent in frame['entities'][str(pid)]:
                    ship_id = int(ent)
                    ent = frame['entities'][str(pid)][ent]
                    frame_entities[ent['y'], ent['x'], 0] = ent['energy']
                    frame_entities[ent['y'], ent['x'], 1] = int(ent['is_inspired']) # Not used
                    frame_entities[ent['y'], ent['x'], pid+num_features] = 1
                    frame_ship_ids[ent['y'], ent['x']] = ship_id
            entities.append(frame_entities)
            ship_ids.append(frame_ship_ids)

        return np.array(entities), np.array(ship_ids)
    
    def load_energy(self, replay):
        energy = [[f['energy'][y] for y in sorted(f['energy'].keys())] for f in replay['full_frames']]
        return np.array(energy)

    def load_deposited(self, replay):
        deposited = [x['deposited'] for x in replay['full_frames']]
        deposited = [[x[y] if y in x else 0 for y in ['0', '1', '2', '3']] for x in deposited]
        return np.array(deposited)

    def get_training_frames(self, pid=None, pname=None, v=None, include_shift=False):
        # pid OR name. Bot version is option (only when name given)
        # if all None, take random player

        if pid is None and pname is not None:
            for p in self.meta_data['players']:
                if pname == p['name'].split(' ')[0]:
                    pid = p['player_id']
                    break
                    
        assert pid is not None, "Error: player not found in replay"
        
        for player in self.meta_data['game_statistics']['player_statistics']:
            if int(player['player_id']) == int(pid):
                did_win = int(player['rank']) == 1
                break
        
        num_p = int(self.meta_data['number_of_players'])
        if pid >= num_p:
            print(self.path)
        
        map_shape = self.production.shape[1], self.production.shape[2]
        factories = np.zeros((*map_shape, 1), dtype=np.float32)
        for fx, fy in self.factories:
            factories[fy, fx] = -1 # Assume all are enemies

        # Then set for the player of interest
        fx, fy = self.factories[pid]
        factories[fy, fx] = 1

        # normalize some of the arrays
        production = (self.production)/1000. # Guessing on norm values for now
        #production = (self.production - 119.)/124.
        
        entities = self.entities.copy()
        try:
            my_ships = entities[:, :, :, 2+pid].copy()
        except:
            print(self.path)
        
        enemy_ship_counts = np.sum(entities[:, :, :, 2:].copy().astype(np.float32), (1, 2))
        enemy_ship_counts = enemy_ship_counts[:, [x for x in range(num_p) if x != pid]]

        has_ship = np.sum(entities[:, :, :, 2:].copy().astype(np.float32), -1)
        
        has_ship_mask = has_ship.copy()
        
        # Convert enemy ships
        has_ship = -1 * has_ship
        has_ship[my_ships>0.5] = 1
        
        # Normalize
        entity_energies = (entities[:, :, :, 0].copy().astype(np.float32))/1000.
        entity_energies *= has_ship_mask
        
        ship_is_full = (entity_energies > 0.9999) * (my_ships>0.5)

        has_ship = np.expand_dims(has_ship, -1)
        entity_energies = np.expand_dims(entity_energies, -1)

        dropoffs = self.dropoffs
        has_dropoff = np.zeros((production.shape[0], *map_shape, 1), dtype=np.float32)
        assert len(dropoffs) == len(production)
        for ix, ds in enumerate(dropoffs):
            for d in ds:
                x, y, oid = d
                v = 1. if oid == pid else -1.
                has_dropoff[ix, y, x] = v

        # factories need to be duplicated across frames
        factories = np.repeat(np.expand_dims(factories, 0), production.shape[0], 0)

        production = np.expand_dims(production, -1)

        ship_is_full = np.expand_dims(ship_is_full, -1)

        ship_id_feat = self.ship_ids * (my_ships>0.5)/50.

        assert ship_id_feat.shape[0] == production.shape[0]

        ship_id_feat = np.expand_dims(ship_id_feat, -1)

        frames = np.concatenate([production, has_ship, entity_energies, factories, has_dropoff, ship_is_full, ship_id_feat], axis=-1)

        # Note: 'no-moves' do not need explicit assignment since 0 means 'still'
        # and the arrays are initialled with zero.
        
        moves = self.moves[:, :, :, pid]

        if include_shift:
            frames, moves, shift = self.center_frames(frames, moves, include_shift) # TODO: Double check that moves are adjusted properly
        else:
            frames, moves = self.center_frames(frames, moves) # TODO: Double check that moves are adjusted properly
        
        if False: # old
            will_have_ship = frames[:, :, :, 1:2] > 0.5 # TODO: resolve generate first, below
            will_have_ship = will_have_ship[1:] # TODO: also resolve these moves in order of confidence
            has_shipyard_or_dropoff = (frames[:1, :, :, 3:4] > 0.5).astype('float32') + (has_dropoff[-1:] > 0.5).astype('float32')
            has_shipyard_or_dropoff = has_shipyard_or_dropoff > 0.5
            will_have_ship = np.concatenate([will_have_ship, has_shipyard_or_dropoff], 0)
            will_have_ship = will_have_ship.astype('uint8')#.astype('float32')
            #will_have_ship = np.expand_dims(will_have_ship, -1)
            will_have_ship = np.squeeze(will_have_ship, -1)
            
        will_have_ship = np.zeros(frames[:, :, :, :5].shape, dtype=np.uint8)
        #converted_moves = np.where(np.logical_and(moves >= 0, moves <= 4))
        will_have_ship[:, :, :, 0] = moves == 0
        will_have_ship[:, :, :, 1] = np.roll(moves == 1, -1, 1)
        will_have_ship[:, :, :, 2] = np.roll(moves == 2,  1, 2)
        will_have_ship[:, :, :, 3] = np.roll(moves == 3,  1, 1)
        will_have_ship[:, :, :, 4] = np.roll(moves == 4, -1, 2)
        
        will_have_ship = np.sum(will_have_ship, -1)
        #print(np.max(will_have_ship))
        will_have_ship = (will_have_ship > 0.5).astype('uint8')

        #frames, my_ships, moves = self.pad_replay(frames, moves)
        
        generate = self.generate[:, pid]
        my_energy = self.energy[:-1, pid] # -1 because I don't need final state here
        
        can_afford_both = my_energy > 4999.
        can_afford_drop = my_energy > 3999.
        can_afford_ship = my_energy > 999.

        mask = np.ones((my_energy.shape[0], self.energy.shape[-1]), np.bool)
        mask[:, pid] = 0

        opponent_energy = self.energy[:-1][mask]
        
        opponent_energy = opponent_energy.reshape((mask.shape[0], -1))
        
        assert opponent_energy.shape[1] == self.energy.shape[1] - 1
        
        map_size_ix = MAP_SIZES.index(frames.shape[1])
        map_size = np.zeros((len(MAP_SIZES),), dtype=np.float32)
        map_size[map_size_ix] = 1.

        my_halite = np.log10(my_energy/1000. + 1)
        my_halite = np.expand_dims(my_halite, -1)
        enemy_halite = np.log10(opponent_energy/1000. + 1)
        _halite_diff = np.expand_dims(my_energy, -1) - opponent_energy
        halite_diff = np.sign(_halite_diff) * np.log10(np.absolute(_halite_diff)/1000. + 1)

        # diff between opponents will also be useful for RL later
        
        can_afford = np.stack([can_afford_ship, can_afford_drop, can_afford_both], -1)
        
        turns_left = np.array(list(range(can_afford.shape[0]-1, -1, -1)))
        turns_left = turns_left/200. - 1.

        turns_left = np.expand_dims(turns_left, -1)
        
        num_opponents = 0 if num_p == 2 else 1
        
        #num_opponent_ships = np.sum(has_ship < 0, axis=(1, 2))/50.
        num_opponent_ships = enemy_ship_counts/50.
        num_my_ships = np.sum(has_ship > 0, axis=(1, 2))/50.
        
        meta_features = np.array(list(map_size) +  [num_opponents])

        assert meta_features.shape[0] == 6
        
        meta_features = np.expand_dims(meta_features, 0)
        meta_features = np.tile(meta_features, [enemy_halite.shape[0], 1]) # expand to all frames
        
        opponent_features = [enemy_halite, halite_diff, num_opponent_ships]
        
        opponent_features = np.stack(opponent_features, -1)
        
        if opponent_features.shape[1] == 1:
            opponent_features = np.pad(opponent_features, ((0,0), (0,2), (0,0)), 'constant', constant_values=0)
        
        my_player_features = [my_halite, turns_left, can_afford, num_my_ships]
        
        my_player_features = np.concatenate(my_player_features, -1)

        # you could also add map starting density to that if youre feeling
        # really extreme. high density is such a different game from low.
        # density on map, gini coefficient, log of total halite on map
        # (normalized on per-size basis)

        my_player_features = np.concatenate([my_player_features, meta_features], -1)
        
        should_construct = np.sum((moves[:, :, :] > 4.5).astype('float32'), (1,2)) > 0.5
        should_construct = should_construct.astype('float32')

        did_win = [did_win for _ in range(should_construct.shape[0])]
        did_win = np.array(did_win).astype('float32')

        if include_shift:
            return frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win, shift
        else:
            return frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win

    def center_frames(self, frames, moves=None, include_shift=False):
        my_factory = frames[0, :, :, 3] > 0.5
        pos = np.squeeze(np.where(my_factory>0))
        expected_pos = np.squeeze(my_factory.shape)//2 # Assuming always square
        shift = expected_pos - pos
        frames = np.roll(frames, shift[0], axis=1)
        frames = np.roll(frames, shift[1], axis=2)
        
        if moves is not None and not include_shift:
            moves = np.roll(moves, shift[0], axis=1)
            moves = np.roll(moves, shift[1], axis=2)
            return frames, moves
        
        elif moves is not None and include_shift:
            moves = np.roll(moves, shift[0], axis=1)
            moves = np.roll(moves, shift[1], axis=2)
            return frames, moves, shift
        
        if include_shift:
            return frames, shift
        else:
            return frames

    def pad_replay_old(self, frames, moves=None, include_padding=False):
        
        map_size = frames.shape[1]
    
        my_ships = (frames[:, :, :, 1] > 0.5).astype(np.float32)
        zeros = np.zeros(my_ships.shape, dtype=np.float32)
    
        # Let's do full padding (by reflection)
        frames = np.concatenate([frames, frames, frames], axis=1)
        frames = np.concatenate([frames, frames, frames], axis=2)
        
        total_x_left_padding = map_size
        total_x_right_padding = map_size
        total_y_left_padding = map_size
        total_y_right_padding = map_size
        
        if moves is not None:
            moves = np.concatenate([moves, moves, moves], axis=1)
            moves = np.concatenate([moves, moves, moves], axis=2)
        
        my_ships = np.concatenate([zeros, my_ships, zeros], axis=1)
        zeros = np.concatenate([zeros, zeros, zeros], axis=1)
        my_ships = np.concatenate([zeros, my_ships, zeros], axis=2)
        zeros = np.concatenate([zeros, zeros, zeros], axis=2)
        
        # Ensure all get padded to the same max dim
        #max_dim = 192
        max_dim = 128 # For easier u-net implementation
        if frames.shape[1] != max_dim:
            pad_y1 = (max_dim - frames.shape[1])//2
            pad_y2 = (max_dim - frames.shape[1]) - pad_y1
            frames = np.concatenate([frames[:, -pad_y1:], frames, frames[:, :pad_y2]], axis=1)
            my_ships = np.concatenate([zeros[:, -pad_y1:], my_ships, zeros[:, :pad_y2]], axis=1)
            zeros = np.concatenate([zeros[:, -pad_y1:], zeros, zeros[:, :pad_y2]], axis=1)
            
            total_y_left_padding += pad_y1
            total_y_right_padding += pad_y2
            
            if moves is not None:
                moves = np.concatenate([moves[:, -pad_y1:], moves, moves[:, :pad_y2]], axis=1)
            
            pad_x1 = (max_dim - frames.shape[2])//2
            pad_x2 = (max_dim - frames.shape[2]) - pad_x1
            frames = np.concatenate([frames[:, :, -pad_x1:], frames, frames[:, :, :pad_x2]], axis=2)
            my_ships = np.concatenate([zeros[:, :, -pad_x1:], my_ships, zeros[:, :, :pad_x2]], axis=2)
            
            total_x_left_padding += pad_x1
            total_x_right_padding += pad_x2
        
            if moves is not None:
                moves = np.concatenate([moves[:, :, -pad_x1:], moves, moves[:, :, :pad_x2]], axis=2)

        if moves is not None:
            return frames, my_ships.astype('uint8'), moves.astype('uint8')
        
        padding = total_x_left_padding, total_x_right_padding, total_y_left_padding, total_y_right_padding

        if include_padding:
            return frames, my_ships, padding
        else:
            return frames, my_ships

    def pad_replay(self, frames, moves=None, include_padding=False, will_have_ship=None, m_weights=None):
        
        map_size = frames.shape[2]
    
        my_ships = (frames[:, :, :, :, 1] > 0.5).astype(np.float32)
        zeros = np.zeros(my_ships.shape, dtype=np.float32)
    
        total_x_left_padding = 0
        total_x_right_padding = 0
        total_y_left_padding = 0
        total_y_right_padding = 0

        # Ensure all get padded to the same max dim
        #max_dim = 192
        max_dim = 128 # For easier u-net implementation
        while frames.shape[2] != max_dim:
            pad_y1 = (min(max_dim, frames.shape[2]*3) - frames.shape[2])//2
            pad_y2 = (min(max_dim, frames.shape[2]*3) - frames.shape[2]) - pad_y1
            frames = np.concatenate([frames[:, :, -pad_y1:], frames, frames[:, :, :pad_y2]], axis=2)

            my_ships = np.concatenate([zeros[:, :, -pad_y1:], my_ships, zeros[:, :, :pad_y2]], axis=2)
            if m_weights is not None:
                m_weights = np.concatenate([zeros[:, :1, -pad_y1:], m_weights, zeros[:, :1, :pad_y2]], axis=2)
            zeros = np.concatenate([zeros[:, :, -pad_y1:], zeros, zeros[:, :, :pad_y2]], axis=2)
            
            total_y_left_padding += pad_y1
            total_y_right_padding += pad_y2
            
            if moves is not None:
                moves = np.concatenate([moves[:, :, -pad_y1:], moves, moves[:, :, :pad_y2]], axis=2)
            
            if will_have_ship is not None:
                will_have_ship = np.concatenate([will_have_ship[:, :, -pad_y1:], will_have_ship, will_have_ship[:, :, :pad_y2]], axis=2)
            
            pad_x1 = (min(max_dim, frames.shape[3]*3) - frames.shape[3])//2
            pad_x2 = (min(max_dim, frames.shape[3]*3) - frames.shape[3]) - pad_x1
            frames = np.concatenate([frames[:, :, :, -pad_x1:], frames, frames[:, :, :, :pad_x2]], axis=3)

            my_ships = np.concatenate([zeros[:, :, :, -pad_x1:], my_ships, zeros[:, :, :, :pad_x2]], axis=3)
            if m_weights is not None:
                m_weights = np.concatenate([zeros[:, :1, :, -pad_x1:], m_weights, zeros[:, :1, :, :pad_x2]], axis=3)
            zeros = np.concatenate([zeros[:, :, :, -pad_x1:], zeros, zeros[:, :, :, :pad_x2]], axis=3)
            
            total_x_left_padding += pad_x1
            total_x_right_padding += pad_x2
        
            if moves is not None:
                moves = np.concatenate([moves[:, :, :, -pad_x1:], moves, moves[:, :, :, :pad_x2]], axis=3)
                
            if will_have_ship is not None:
                will_have_ship = np.concatenate([will_have_ship[:, :, :, -pad_x1:], will_have_ship, will_have_ship[:, :, :, :pad_x2]], axis=3)
    
        padding = total_x_left_padding, total_x_right_padding, total_y_left_padding, total_y_right_padding
        
        if moves is not None and will_have_ship is not None and not include_padding and m_weights is not None:
            return frames, my_ships.astype('uint8'), moves.astype('uint8'), will_have_ship.astype('uint8'), m_weights.astype('uint8')
                
        if moves is not None and will_have_ship is not None and not include_padding:
            return frames, my_ships.astype('uint8'), moves.astype('uint8'), will_have_ship.astype('uint8')
        
        if moves is not None and will_have_ship is not None and include_padding:
            return frames, my_ships.astype('uint8'), moves.astype('uint8'), will_have_ship.astype('uint8'), padding

        if moves is not None:
            return frames, my_ships.astype('uint8'), moves.astype('uint8')
        
        

        if include_padding:
            return frames, my_ships, padding
        else:
            return frames, my_ships
    
    def reverse_padding(self, true_size):
        pass

    def aug(move, frame, iter, frames_padding):
        """
        Augment the data using rotations and mirroring.
        """
        should_mirror = iter>3
        
        frames_padding = np.array([[frames_padding[0][0], frames_padding[0][1]],
                                    [frames_padding[1][0], frames_padding[1][1]]])
        

        num_rotate = iter%4
        
        if should_mirror:
            frame = np.flip(frame, 0)
            move = np.flip(move, 0)
        
            frames_padding = np.array([[frames_padding[0][1], frames_padding[0][0]],
                                        [frames_padding[1][0], frames_padding[1][1]]])
        
        frame = np.rot90(frame, num_rotate)
        move = np.rot90(move, num_rotate)
        
        for _ in range(num_rotate):
            frames_padding = np.array([[frames_padding[1][1], frames_padding[1][0]],
                                       [frames_padding[0][0], frames_padding[0][1]]])
        
        # We need to change the move values to match the augmentation
        # STILL will not change; this is simply reordering the axis
        # Current axis: N:1, E:2, S:3, W:4
        if should_mirror:
            move = move[:, :, (0, 3, 2, 1, 4)]

        shift = np.roll(move[:, :, 1:].copy(), -1*num_rotate, 2)
        move[:, :, 1:] = shift

        return move, frame, frames_padding

