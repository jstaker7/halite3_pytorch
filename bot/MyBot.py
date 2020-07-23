#!/usr/bin/env python3

LOCAL = True

import os
import sys
#import time
if LOCAL:
    import logging
import json
#import arrow

import numpy as np

import torch

cd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cd)
from data_utils import center_frames, pad_replay
from architecture import Model

window_size = 1

model = Model(1, window_size)

# Required inputs:
# production, has_ship, entity_energies, factories, has_dropoff
# can_afford
# turns_left

MAP_SIZES = [32, 40, 48, 56, 64]

opponents = {}

def update_frame(frame, num_players, my_id, map_dim, max_turns):

    global opponents

    # Some channels get refreshed entirely
    frame[:, :, [1,2,4,5,6]] = 0

    turn_number = input()
    
    turns_left_raw = float(max_turns) - float(turn_number)
    
    turns_left = (turns_left_raw)/200. - 1.
    
    if LOCAL:
        logging.info(turns_left_raw)

    turns_left = np.expand_dims(turns_left, 0)
    
    my_halite = None
    my_ships = None
    
    my_dropoffs = []
    enemy_dropoffs = []

    ship_locs = {}

    for _ in range(num_players):
        player, num_ships, num_dropoffs, halite = [int(x) for x in input().split()]
        
        ships = [[int(x) for x in input().split()] for _ in range(num_ships)]
        dropoffs = [[int(x) for x in input().split()] for _ in range(num_dropoffs)]
        
        if player == my_id:
            my_halite = halite
            my_ships = ships
        else:
            if player not in opponents:
                opponents[player] = {'halite': halite, 'num_ships': num_ships}
            else:
                opponents[player]['halite'] = halite
                opponents[player]['num_ships'] = num_ships

        for ship in ships:
            id, x, y, h = ship
            ship_locs[(player, id)] = (y, x)
            frame[y, x, 2] = h
            if player == my_id:
                frame[y, x, 1] = 1.
                frame[y, x, 5] = float(h > 999)
                frame[y, x, 6] = float(id)/50.
            else:
                frame[y, x, 1] = -1.
    
        for dropoff in dropoffs:
            id, x, y = dropoff
            if player == my_id:
                frame[y, x, 4] = 1.
                my_dropoffs.append((y, x))
            else:
                frame[y, x, 4] = -1.
                enemy_dropoffs.append((y, x))

    
    for _ in range(int(input())):
        x, y, h = [int(x) for x in input().split()]
        frame[y, x, 0] = h

    assert my_halite is not None

    can_afford_both = my_halite > 4999.
    can_afford_drop = my_halite > 3999.
    can_afford_ship = my_halite > 999.

    can_afford = np.stack([can_afford_ship, can_afford_drop, can_afford_both], -1)
    
    del can_afford_ship
    del can_afford_drop
    del can_afford_both

    op_ids = sorted(list(opponents.keys()))

    opponent_energy = np.array([opponents[x]['halite'] for x in op_ids])

    opponent_energy = opponent_energy.reshape((1, -1))

    # TODO: This only needs to be computed once
    map_size_ix = MAP_SIZES.index(map_dim)
    map_size = np.zeros((len(MAP_SIZES),), dtype=np.float32)
    map_size[map_size_ix] = 1.

    _my_halite = int(my_halite)

    my_halite = np.log10(_my_halite/1000. + 1)
    my_halite = np.expand_dims(my_halite, -1)
    enemy_halite = np.log10(opponent_energy/1000. + 1)
    _halite_diff = np.expand_dims(_my_halite, -1) - opponent_energy
    halite_diff = np.sign(_halite_diff) * np.log10(np.absolute(_halite_diff)/1000. + 1)

    num_opponents = 0 if len(op_ids) == 1 else 1

    enemy_ship_counts = np.array([opponents[x]['num_ships'] for x in op_ids])

    num_opponent_ships = enemy_ship_counts/50.
    num_opponent_ships = np.expand_dims(num_opponent_ships, 0)
    num_my_ships = [len(my_ships)/50.]
        
    meta_features = np.array(list(map_size) +  [num_opponents])

    assert meta_features.shape[0] == 6

    meta_features = np.expand_dims(meta_features, 0)
    meta_features = np.tile(meta_features, [enemy_halite.shape[0], 1])

    opponent_features = [enemy_halite, halite_diff, num_opponent_ships]
    
    opponent_features = np.stack(opponent_features, -1)
    if opponent_features.shape[1] == 1:
        opponent_features = np.pad(opponent_features, ((0,0), (0,2), (0,0)), 'constant', constant_values=0)
    my_player_features = [my_halite, turns_left, can_afford, num_my_ships]

    my_player_features = np.concatenate(my_player_features, -1)

    my_player_features = np.expand_dims(my_player_features, 0)

    my_player_features = np.concatenate([my_player_features, meta_features], -1)

    return frame, turns_left_raw, my_player_features, opponent_features, my_ships, _my_halite, my_dropoffs, enemy_dropoffs, ship_locs

def get_initial_data():
    raw_constants = input()

    constants = json.loads(raw_constants)
    
    max_turns = constants['MAX_TURNS'] # Only one I think we need
    
    del constants

    num_players, my_id = [int(x) for x in input().split()]
    
    player_tups = []
    for player in range(num_players):
        p_tup = map(int, input().split())
        player_tups.append(p_tup)
    
    map_width, map_height = map(int, input().split())
    map_dim = map_width # Assuming square maps (to keep things simple)

    game_map = []
    for _ in range(map_dim):
        row = [int(x) for x in input().split()]
        game_map.append(row)

    halite = np.array(game_map)
    
    del game_map

    return max_turns, num_players, my_id, halite, player_tups, map_dim

valid_moves = ['o', 'n', 'e', 's', 'w', 'c']
move_shifts = [(0,0), (-1,0), (0,1), (1,0), (0,-1), (0,0)]

#assert os.path.exists(os.path.join(cd, "model.ckpt.meta"))
#assert os.path.exists(os.path.join(cd, "model.ckpt.data-00000-of-00001"))

#logging.info("Starting session...")
# Load the model

# TODO: make flexible to support both CPU and GPU device types
model.load_state_dict(torch.load(os.path.join(cd, "model.pt"), map_location='cpu'))
model.eval()

# TODO: Also load player model

max_turns, num_players, my_id, halite, player_tups, map_dim = get_initial_data()

if LOCAL:
    logging.basicConfig(filename='/path/logs/{}-bot.log'.format(my_id),
                                filemode='w',
                                #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                #datefmt='%H:%M:%S',
                                level=logging.DEBUG)

frame = np.zeros((map_dim, map_dim, 7), dtype=np.float32)
frame[:, :, 0] = halite.copy()

del halite

enemy_shipyards = []

for player in player_tups:
    player, shipyard_x, shipyard_y = player
    if int(player) == my_id:
        frame[shipyard_y, shipyard_x, 3] = 1.
        my_shipyard = shipyard_y, shipyard_x
    else:
        frame[shipyard_y, shipyard_x, 3] = -1.
        enemy_shipyards.append((shipyard_y, shipyard_x))

del player_tups

MAX_DIM = 64

# Prime the system
f = np.zeros((1, 1, MAX_DIM, MAX_DIM, 7), dtype=np.float32)
p = np.zeros((1, 1, 12), dtype=np.float32)
o = np.zeros((1, 1, 3, 3), dtype=np.float32)
m = np.zeros((1, 1, MAX_DIM, MAX_DIM), dtype=np.float32)

_f = np.concatenate([f] * window_size, 1)
_p = np.concatenate([p] * window_size, 1)
_o = np.concatenate([o] * window_size, 1)
_m = np.concatenate([m] * window_size, 1)

for _ in range(1):
    _ = model(_f, _p, _o, moves=_m, train=False, num_players=1, valid=False)

# Send name
print("jstaker7", flush=True)
#print([x.name for x in local_device_protos if x.device_type == 'GPU'])
#sys.stdout.flush()

moves_raw = np.zeros((1, 1, map_dim, map_dim), dtype=np.float32)
move_zero = np.zeros((1, 1, MAX_DIM, MAX_DIM), dtype=np.float32)

frame_history = [f for _ in range(window_size-1)]
my_player_features_history = [p for _ in range(window_size-1)]
opponent_features_history = [o for _ in range(window_size-1)]
moves_history = [move_zero for _ in range(window_size - 1)]
last_ship_locs = {}

for turn in range(999999):
    frame, turns_left, my_player_features, opponent_features, my_ships, my_halite, my_dropoffs, enemy_dropoffs, ship_locs = update_frame(frame, num_players, my_id, map_dim, max_turns)
    
    
    
    ship_moves = {}
    moves_raw.fill(0) # Reset (in place)
    
    for ship in last_ship_locs:
        if ship in ship_locs:
            # easy (the ship persisted between frames)
            ly, lx = last_ship_locs[ship]
            y, x, = ship_locs[ship]
            # Note: these also include cases where the map wraps
            if x < lx or lx - x < -2:
                m = valid_moves.index('w')
            elif x > lx or lx - x > 2:
                m = valid_moves.index('e')
            elif y < ly or ly - y < -2:
                m = valid_moves.index('n')
            elif y > ly or ly - y > 2:
                m = valid_moves.index('s')
            elif x == lx and y == ly:
                m = valid_moves.index('o')
            else:
                assert False
            ship_moves[ship] = m
            moves_raw[:, :, ly, lx] = m
        else:
            # hard (crashed or made a dropoff)
            pass

    last_ship_locs = ship_locs
    
    _frame = np.expand_dims(frame.copy(), 0) # Expects batch dim
    
    cost_to_move = np.floor(_frame[0, :, :, 0].copy() * 0.1)
    can_afford_to_move = (_frame[0, :, :, 2].copy() - cost_to_move) >= -1e-13

    _frame[:, :, :, 0] =  (_frame[:, :, :, 0])/1000.
    _frame[:, :, :, 2] =  (_frame[:, :, :, 2])/1000.
    
    has_my_ships = _frame[0, :, :, 1].copy() > 0.5
    
    # Center
    _frame, prev_moves, shift = center_frames(_frame, moves_raw, True)
    
    _frame, prev_moves, padding = pad_replay(_frame, prev_moves)

    moves_history.append(prev_moves.copy())
    
    lxp, rxp, lyp, ryp = padding
    
    _frame = np.expand_dims(_frame, 1)
    my_player_features = np.expand_dims(my_player_features, 1)
    opponent_features = np.expand_dims(opponent_features, 1)
    
    frame_history.append(_frame)
    my_player_features_history.append(my_player_features)
    opponent_features_history.append(opponent_features)

    _frame = np.concatenate(frame_history[-window_size:], 1)
    my_player_features = np.concatenate(my_player_features_history[-window_size:], 1)
    opponent_features = np.concatenate(opponent_features_history[-window_size:], 1)
    
    if window_size == 1:
        moves = move_zero
    else:
        moves = np.concatenate(moves_history[-(window_size-1):] + [move_zero], 1)
    
    # While debugging
    #moves = moves * 0

    # TODO: Pad, keep track of where the ships are

    with torch.no_grad():
        mo, go, m_probs, latent = model(_frame,
                                        my_player_features,
                                        opponent_features,
                                        train=False,
                                        num_players=1,
                                        valid=False,
                                        moves=moves)

    del _frame
    del my_player_features
    del opponent_features

    #logging.info(wo)

    if False:#LOCAL:
        which = np.argmax(np.squeeze(wo))
    else:
        which = 0

    #logging.info(which)

    mo = mo[which]
    go = go[which]

    if False:#LOCAL:
        ho = ho[which]
        bo = np.squeeze(bo[which])

    # TODO: Also get the game state to determine the player to use

    if False:
        logging.info(go)

    go = np.squeeze(go) > 0 # Raw number, 0 is sigmoid()=0.5 # TODO: double check this
#    logging.info(mo.shape)
#    logging.info(lyp)
#    logging.info(ryp)
#    logging.info(lxp)
#    logging.info(rxp)
    if lyp != 0 and ryp != 0 and lxp != 0 and rxp != 0:
        mo = mo[0, :, lyp:-ryp, lxp:-rxp] # reverse pad
    else:
        mo = mo[0]

    mo = np.transpose(mo, (1, 2, 0))

    if False:#LOCAL:
        ho = ho[0, lyp:-ryp, lxp:-rxp, 0] # reverse pad, remove extra dim
    
    #highest_confidence = np.max(mo, -1)
    #highest_confidence_loc = np.argmax(highest_confidence, [0, 1])

    mo = np.roll(mo, -shift[0], axis=0) # reverse center
    mo = np.roll(mo, -shift[1], axis=1) # reverse center

    if False:#LOCAL:
        ho = np.roll(ho, -shift[0], axis=0) # reverse center
        ho = np.roll(ho, -shift[1], axis=1) # reverse center
    
    # Ensure ships on shipyard/dropoff are forced to move
    # NOTE: Sampling will not work if this is enabled
#        for loc in my_dropoffs + [my_shipyard]:
#            mo[loc[0], loc[1]] += 1000.
#            mo[loc[0], loc[1], 0] -= 1000.

    if False:#LOCAL:
        raw_h_moves = [ho,
                       np.roll(ho,  1, 0),
                       np.roll(ho, -1, 1),
                       np.roll(ho, -1, 0),
                       np.roll(ho,  1, 1)]

        raw_h_moves = np.stack(raw_h_moves, -1)

        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / np.expand_dims(e_x.sum(axis=-1), -1)
        
        h_moves = softmax(raw_h_moves)
        h_moves *= np.expand_dims(np.sum(mo[:, :, :5], -1), -1)
    
    combined = mo.copy()
#        combined[:, :, 0] *= ho
#        combined[:, :, 1] *= np.roll(ho,  1, 0)
#        combined[:, :, 2] *= np.roll(ho, -1, 1)
#        combined[:, :, 3] *= np.roll(ho, -1, 0)
#        combined[:, :, 4] *= np.roll(ho,  1, 1)
#        
#        combined[:, :, :5] *= np.expand_dims(np.max(mo[:, :, :5], -1)/np.max(combined[:, :, :5], -1), -1)

    #combined[:, :, :5] += h_moves
    #combined[:, :, 5] *= 2
    #combined /= 2
    
    #combined[:, :, 5] *= 1.25

    mo = mo * np.expand_dims(has_my_ships, -1)
    combined = combined * np.expand_dims(has_my_ships, -1)
    
    if False:#LOCAL:
        h_moves = h_moves * np.expand_dims(has_my_ships, -1)
    
        if False:
            y_ = 32
            x_ = 47
    #        if turns_left < 400:
    #            mo[19:22, 10:13, 0] -= 1000
#            logging.info((mo[19:22, 10:13]*100).astype('uint8'))
#            logging.info((ho[19:22, 10:13]*100).astype('uint8'))
#            logging.info((combined[19:22, 10:13]*100).astype('uint8'))

            logging.info((mo[y_-1:y_+2, x_-1:x_+2]*100).astype('int32'))
            logging.info((ho[y_-1:y_+2, x_-1:x_+2]*100).astype('int32'))
            logging.info((combined[y_-1:y_+2, x_-1:x_+2]*100).astype('int32'))
            logging.info((h_moves[y_-1:y_+2, x_-1:x_+2]*100).astype('int32'))

    #logging.info(bo)

    mo = combined

    if False:#LOCAL:
        _ho = ho.copy()

        ho = ho[:, :] > 0

        if np.sum(ho) > 0:
            h_y, h_x = np.where(ho)
            have_ship = list(zip(h_y, h_x))
        else:
            have_ship = []

        # Sort by confidence (important for crowded areas)
        _have_ship = []
        for hs in have_ship:
            y, x = hs
            _have_ship.append((y, x, _ho[y, x]))
        have_ship = _have_ship

        have_ship = sorted(have_ship, key=lambda x: -x[2])

        del ho
        del _ho

    # turn my_ships into a dict for easier tiered processing
    # TODO: move this to the function above
    _my_ships = {}
    for ship in my_ships:
        _my_ships[ship[0]] = ship

    my_ships = _my_ships

    constructed = False

    commands = []

    # Attempt to reduce collisions. This is a heuristic that I'd love to
    # be handled through the learning process.
    is_taken = np.zeros(mo.shape[:2], dtype=np.bool)

#        if bo > 0 and my_halite >= 4000:
#            best_ship = None
#            best_conf = -9999
#            ix = 5
#            for ship in my_ships:
#                id, x, y, h = ship
#                conf = mo[y, x, ix].copy()
#                if conf > best_conf:
#                    best_conf = conf
#                    best_ship = id
#
#            if best_ship is not None:
#                id, x, y, h = my_ships[best_ship]
#                mo[y, x, ix] += 1000

#        for loc in have_ship:
#            if not loc == (my_shipyard[0], my_shipyard[1]):
#                continue
#            for ship in my_ships:
#                id, x, y, h = my_ships[ship]
#                for ix, (y_s, x_s) in enumerate(move_shifts):
#                    s_loc = (y + y_s)%map_dim, (x + x_s)%map_dim
#                    if (loc[0], loc[1]) == s_loc:
#                        mo[y, x, ix] += 1000

    assert mo.shape[0] == mo.shape[1] == map_dim

    already_taken = enemy_shipyards + enemy_dropoffs + my_dropoffs + [my_shipyard]
    
    # Filter out moves when they can't be afforded
    # Also focus on ships nearest dropoffs
    _my_ships = []
    for ship in my_ships:
        id, x, y, h = my_ships[ship]
        manhat_dists = [abs(k[0]-y) + abs(k[1]-x) for k in my_dropoffs + [my_shipyard]]
        min_md = int(min(manhat_dists))
        a = can_afford_to_move[y, x]
        c = max(mo[y, x])
        #if min_md <= 4:
        c += 20000 - 25*min_md
        _my_ships.append((id, x, y, h, a, c))

    my_ships = sorted(_my_ships, key=lambda x: (x[4], -x[5], x[0]))

    for ship in my_ships:
        id, x, y, h, a, _ = ship
        
        #ranked_choices = sorted(list(zip(valid_moves, mo[y, x])), key=lambda x: x[1], reverse=True)
        
        probs = mo[y, x].copy()
        
        logging.info(probs)
        
        #probs[-1] += 250

        #probs = [np.random.uniform(0, i) for i in probs] # TODO: Do I really want to sample moves at this point?
        ranked_choices = sorted(list(zip(valid_moves, probs)), key=lambda x: x[1], reverse=True)
        
        for choice in ranked_choices:
            m, _ = choice # don't need probability for now
        
            # TODO: only allow 1 construction per turn?
            if m == 'c' and (my_halite) >= 4000 and (y, x) not in already_taken and not constructed: # TODO: halite on cell and in ship can technically be included; already taken is a problem here; should construct first #  and choice != ranked_choices[-1]
                move_cmd = "c {}".format(id)
                commands.append(move_cmd)
                my_halite -= 4000
                constructed = True
                break
            elif m == 'c':
                continue
            
            if m != 'o' and not a:
                continue # can't afford the move
            
            move_ix = valid_moves.index(m)
            y_s, x_s = move_shifts[move_ix]
            loc = (y + y_s)%map_dim, (x + x_s)%map_dim

            if not is_taken[loc[0], loc[1]]:
                move_cmd = "m {} {}".format(id, m)
                commands.append(move_cmd)
                is_taken[loc[0], loc[1]] = 1
                break

        if turns_left < 40:
            for loc in my_dropoffs + [my_shipyard]:
                is_taken[loc[0], loc[1]] = 0

    if not is_taken[my_shipyard[0], my_shipyard[1]] and go and my_halite >= 1000:
        commands.append("g")
        my_halite -= 1000

    print(" ".join(commands))
    sys.stdout.flush()


