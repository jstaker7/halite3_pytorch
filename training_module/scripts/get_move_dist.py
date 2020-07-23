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

if not os.path.exists(replay_root):
    replay_root = '/path/replays'

with gzip.open(os.path.join(replay_root, 'INDEX.pkl'), 'rb') as infile:
    master_index = pickle.load(infile)

pname = 'teccles'
versions = [166]

keep = set()
for rp in master_index:
    for p in master_index[rp]['players']:
        name, _version = p['name'].split(' v')
        version = int(_version.strip())
        if pname == name and version in versions:
            keep.add(rp)
            break

print('{} replays'.format(len(keep)))

counts = {}
running_totals = {}
running_means = {}
running_count = {}

for which_game in keep:
    path = os.path.join(replay_root, '{}', '{}')
    day = which_game.replace('ts2018-halite-3-gold-replays_replay-', '').split('-')[0]
    path = path.format(day, which_game)

    game = Game()
    try:
        game.load_replay(path)
    except:
        continue

    frames, moves, generate, _, opponents, _, _, _ = game.get_training_frames(pname=pname)
    num_opponents = np.sum(opponents[0, :, 0] > 0)
    map_size = moves.shape[1]
    length = moves.shape[0] # Corner cases where games end early

    my_ships = (frames[:, :, :, 1] > 0.5)

    key = str(num_opponents) + '_' + str(map_size) + '_' + str(length)
    if key not in counts:
        counts[key] = {}

    #totals = np.array([v for _, v in counts[key].items()]).sum(0)

    m = np.ma.array(moves, mask=~my_ships)

    o = (m == 0).astype('float32').sum(1).sum(1)
    n = (m == 1).astype('float32').sum(1).sum(1)
    e = (m == 2).astype('float32').sum(1).sum(1)
    s = (m == 3).astype('float32').sum(1).sum(1)
    w = (m == 4).astype('float32').sum(1).sum(1)
    c = (m == 5).astype('float32').sum(1).sum(1)

    combined = np.array([o, n, e, s, w, c])

    totals = combined.sum(0)

    if key not in running_totals:
        running_totals[key] = totals

    totals[totals < 1e-32] = 1e-32 # fill zero move frames

    # Percents instead of ship count so that all frames are weighted equally
    percents = combined/np.expand_dims(totals, 0)

    if key not in running_means:
        running_means[key] = percents
        running_count[key] = 0
    else:
        stacked = np.stack([running_means[key], percents])
        running_count[key] += 1
        running_means[key] = np.average(stacked, 0, [running_count[key], 1])

with open('move_counts.pkl', 'wb') as handle:
    pickle.dump(running_means, handle)
