import copy
import numpy as np
import torch
import os
import gzip
import pickle

from training_module.data_utils import Game
from training_module.architecture import Model

from PIL import Image

replay_root = '/path/replays'

with gzip.open(os.path.join(replay_root, 'INDEX.pkl'), 'rb') as infile:
    master_index = pickle.load(infile)

PLAYERS = [
            {'pname': 'teccles',
             'versions': [166],
             },
]

in_train = set()
in_valid = set()

def filter_replays(pname, versions):
    global in_train
    global in_valid
    keep = []
    for rp in master_index:
        for p in master_index[rp]['players']:
            name, _version = p['name'].split(' v')
            version = int(_version.strip())
            if pname == name and version in versions:
                keep.append(rp)
                break

    train, valid = [], []
    new_keep = []
    for rp in keep:
        if rp in in_train:
            train.append(rp)
        elif rp in in_valid:
            valid.append(rp)
        else:
            new_keep.append(rp)

    _train, _valid = new_keep[:int(len(new_keep)/1.33)], new_keep[int(len(new_keep)/1.33):]

    train += _train
    valid += _valid

    in_train |= set(train)
    in_valid |= set(valid)

    return train, valid

for player in PLAYERS:
    train, valid = filter_replays(player['pname'], player['versions'])
    player['train'] = train
    player['valid'] = valid
    print("{} num train: {} num valid: {}".format(player['pname'], len(train), len(valid)))

del master_index


def render_moves(mo, my_ship, should_arg=True):
    img = np.ones((mo.shape[0], mo.shape[1], 3)) * 255

    if should_arg:
        maxed = np.argmax(mo, -1)
    else:
        maxed = mo

    y, x = np.where((maxed == 0)*my_ship)

    img[y, x, 0] = 255
    img[y, x, 1] = 0
    img[y, x, 2] = 0

    y, x = np.where((maxed == 1)*my_ship)

    img[y, x, 0] = 0
    img[y, x, 1] = 255
    img[y, x, 2] = 0

    y, x = np.where((maxed == 2)*my_ship)

    img[y, x, 0] = 0
    img[y, x, 1] = 0
    img[y, x, 2] = 255

    y, x = np.where((maxed == 3)*my_ship)

    img[y, x, 0] = 255
    img[y, x, 1] = 255
    img[y, x, 2] = 0

    y, x = np.where((maxed == 4)*my_ship)

    img[y, x, 0] = 255
    img[y, x, 1] = 0
    img[y, x, 2] = 255

    y, x = np.where((maxed == 5)*my_ship)

    img[y, x, 0] = 0
    img[y, x, 1] = 0
    img[y, x, 2] = 0

    img = Image.fromarray(img.astype('uint8'))

    img = img.resize((mo.shape[0]*6*2, mo.shape[0]*6*2), Image.NEAREST)

    return img

WINDOW_SIZE = 8

model = Model(1, WINDOW_SIZE)

path = '/path/model.pt'

model.load_state_dict(torch.load(path, map_location='cpu'))
model.eval()

path = os.path.join(replay_root, '20190128', player['train'][0])
print(path)

game = Game()
game.load_replay(path)

frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win, shift = game.get_training_frames(pname='teccles', include_shift=True)


for i in range(frames.shape[0]): # Take a few samples to relieve pre-compute cost

    end_ix = i + 1
    
    s_ix = max(end_ix-WINDOW_SIZE, 0)

    # Avoid GC issues
    _frames = copy.deepcopy(frames[s_ix:end_ix])
    _moves = copy.deepcopy(moves[s_ix:end_ix])
    _generate = copy.deepcopy(generate[s_ix:end_ix])
    _my_player_features = copy.deepcopy(my_player_features[s_ix:end_ix])
    _opponent_features = copy.deepcopy(opponent_features[s_ix:end_ix])
    _will_have_ship = copy.deepcopy(will_have_ship[s_ix:end_ix])
    _should_construct = copy.deepcopy(should_construct[s_ix:end_ix])
    


    _did_win = copy.deepcopy(did_win[0]) # only need 1
    
    ns = _frames.shape[0]
    
    if _frames.shape[0] < WINDOW_SIZE:
        _frames = np.pad(_frames, [(WINDOW_SIZE - ns, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
        _moves = np.pad(_moves, [(WINDOW_SIZE - ns, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
        _generate = np.pad(_generate, [(WINDOW_SIZE - ns, 0)], 'constant', constant_values=0)
        _my_player_features = np.pad(_my_player_features, [(WINDOW_SIZE - ns, 0), (0, 0)], 'constant', constant_values=0)
        _opponent_features = np.pad(_opponent_features, [(WINDOW_SIZE - ns, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
        _will_have_ship = np.pad(_will_have_ship, [(WINDOW_SIZE - ns, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
        _should_construct = np.pad(_should_construct, [(WINDOW_SIZE - ns, 0)], 'constant', constant_values=0)

    shapes = [_frames.shape[0], _moves.shape[0], _generate.shape[0], _my_player_features.shape[0], _opponent_features.shape[0], _will_have_ship.shape[0], _should_construct.shape[0]]

    assert len(set(shapes)) == 1, print(shapes)
    


    frame, my_ship, move, _will_have_ship, padding = game.pad_replay(_frames[None, ...], _moves[None, ...], will_have_ship=_will_have_ship[None, ...], include_padding=True)
    
    lxp, rxp, lyp, ryp = padding

#        if i + 1 == 17:
#            #mo = mo * np.expand_dims(my_ship, -1)
#            #print(mo[31:33, 29:33])
#            #print(_my_player_features)
#            print(_opponent_features)
#            #print(frame[0, 0, 63:65, 61:65])
#            break


    #my_ships = (frame[0, -1, :, :, 1] > 0.5).astype(np.float32)
#    img = render_moves(move[0, -1], my_ship[0, -1], should_arg=False)
#    img.save('/Users/Peace/Desktop/renderings/moves_{}.png'.format(i+1))
#    img = frame[0, -1, :, :, 1]
#    img.save('/Users/Peace/Desktop/renderings/moves_{}.png'.format(i+1))
#    continue

#    img = Image.fromarray((my_ship[0, -1]*255).astype('uint8'))
#    img = img.resize((my_ship.shape[2]*6, my_ship.shape[2]*6), Image.NEAREST)
#    img.save('/Users/Peace/Desktop/renderings/my_ship_{}.png'.format(i+1))
#        img = Image.fromarray((my_ship[0, -1]*255).astype('uint8'))
#        img = img.resize((mo.shape[0]*6, mo.shape[0]*6), Image.NEAREST)
#        img.save('/Users/Peace/Desktop/renderings/frame_{}.png'.format(i+1))



    with torch.no_grad():
        mo, go, m_probs, latent = model(frame,
                                        _my_player_features[None, ...],
                                        _opponent_features[None, ...],
                                        train=False,
                                        num_players=1,
                                        valid=False,
                                        moves=move)
    #print(go)

#        if i == 182:
#            print(move.shape)
#            print(np.sum(move[0, 0] == 4))

    mo = mo[0, 0, :, lyp:-ryp, lxp:-rxp] # reverse pad
    my_ship = my_ship[0, -1, lyp:-ryp, lxp:-rxp] # reverse pad

    mo = np.transpose(mo, (1, 2, 0))

    mo = np.roll(mo, -shift[0], axis=0) # reverse center
    mo = np.roll(mo, -shift[1], axis=1) # reverse center

    my_ship = np.roll(my_ship, -shift[0], axis=0) # reverse center
    my_ship = np.roll(my_ship, -shift[1], axis=1) # reverse center

#        if i == 182:
#            import sys
#            import numpy
#            np.set_printoptions(threshold=sys.maxsize)
#            mo = mo * np.expand_dims(my_ship, -1)
#            print(mo[30, 48])
#            #print(frame)
#            print(np.sum(move[0, 0] == 4))
#            break

    img = render_moves(mo, my_ship)
    img.save('/path/moves_{}.png'.format(i+1))


