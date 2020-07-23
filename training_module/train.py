import pickle
import gzip
import time
from random import shuffle

import os
from threading import Thread
from queue import Queue, PriorityQueue
import copy

import numpy as np

import torch

from training_module.data_utils import Game

from training_module.architecture import Model

np.random.seed(8)

RESTORE = False
RESTORE_WHICH = '4'

batch_size = 7 # these are per map size per player, i.e., effective batch size is larger than this number

# To control amount of mixing (to decorrelate examples) during buffering. Value of 1 means
# strict ordering (no mixing). Larger numbers mix for longer. Keep in mind
# that games are around 400 frames each, so significant mixing may be
# required.
OVERLAP = 1000

# This is the absolute upper bound and the practical size is likely much smaller,
# but as long as we have the memory, we don't care too much, and just set at the
# max to be safe.
MIN_BUFFER_SIZE = OVERLAP + int(batch_size/5) # 5 is number of map types (put in variable?); is this actually right?

max_buffer_size = MIN_BUFFER_SIZE + 10 # Probably don't need much extra here

#WINDOW_SIZE = 4 # Set to None to take all
WINDOW_SIZE = 1

replay_root = '/path/replays'

save_dir = '/path/models/'

if not os.path.exists(save_dir):
    save_dir = '/path/model_temp'

if not os.path.exists(replay_root):
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

mw_dir = '/path/halite3_rl'

if not os.path.exists(mw_dir):
    mw_dir = '/path/'

with open(os.path.join(mw_dir, 'move_counts.pkl'), 'rb') as handle:
    counts = pickle.load(handle)

WEIGHTS = {}
for k, v in counts.items():
    maxed = np.max(v, 0)
    #maxed[maxed < 1e-32] = 1e-32

    #print((1 - v) / still)
    copied = v.copy()
    copied[copied < 1e-32] = 1e-32
    copied = 1/copied
    copied[v < 1e-32] = 0

    WEIGHTS[k] = copied * maxed

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

def batch_prep(buffer, batch_queue, min_buffer_size):
    game = Game()

    while True:
        if buffer.qsize() < min_buffer_size:
            time.sleep(1)
            continue
        
        frames = []
        moves = []
        generates = []
        my_player_features = []
        opponent_features = []
        my_ships = []
        
        will_have_ships = []
        should_constructs = []
        did_wins = []
        
        m_weightss = []
    
        for _ in range(batch_size):
            p, t, pairs = buffer.get()
            for data in pairs:
                
                frame, move, generate, my_player_feature, opponent_feature, will_have_ship, should_construct, did_win, m_weights = data
                
                frame = np.expand_dims(frame, 0)
                move = np.expand_dims(move, 0)
                will_have_ship = np.expand_dims(will_have_ship, 0)
                m_weights = np.expand_dims(np.expand_dims(m_weights, 0), 0)

                frame, my_ship, move, will_have_ship, m_weights = game.pad_replay(frame, move, will_have_ship=will_have_ship, m_weights=m_weights)
                
                frames.append(frame[0])
                moves.append(move[0])
                will_have_ships.append(will_have_ship[0])
                generates.append(generate)
                my_player_features.append(my_player_feature)
                opponent_features.append(opponent_feature)
                my_ships.append(my_ship[0])
                should_constructs.append(should_construct)
                did_wins.append(did_win)
                m_weightss.append(m_weights[0, 0])
    
        pair = np.array(frames), np.array(moves), np.array(generates), np.array(my_player_features), np.array(opponent_features), np.array(my_ships), np.array(will_have_ships), np.array(should_constructs), np.array(did_wins), np.array(m_weightss)
        batch_queue.put(copy.deepcopy(pair))

def buffer(raw_queues, buffer_q, overlap):
    count = 0
    while True:
        #which_queue = np.random.randint(5)
        pairs = [] # We buffer different sizes together to ensure balance
        # TODO: This is actually problematic because each game size starts
        # from the start (too much similarity in batch)
        for which_queue in range(5):
            queue = raw_queues[which_queue]

            pair = queue.get()
            pairs.append(pair)
            
        # Basically shuffles as it goes
        rand_priority = np.random.randint(count, count + overlap)
        buffer_q.put((rand_priority, time.time(), pairs))
        count += 1

default_lengths = {48: 450, 64: 500, 56: 475, 40: 425, 32: 400}
from random import shuffle
def worker(queue, size, pname, keep):
    np.random.seed(size) # Use size as seed
    # Filter out games that are not the right size
    # Note: Replay naming is not consistent (game id was added later)
    s_keep = [x for x in keep if int(x.split('-')[-2]) == size]
    print("{0} {1} maps with size {2}x{2}".format(pname, len(s_keep), size))

    current = s_keep.copy()
    shuffle(current)
    while True:
        #which_game = np.random.choice(s_keep)
        
        which_game = current.pop()
        path = os.path.join(replay_root, '{}', '{}')
        day = which_game.replace('ts2018-halite-3-gold-replays_replay-', '').split('-')[0]
        path = path.format(day, which_game)

        game = Game()
        try:
            game.load_replay(path)
        except:
            continue
    
        frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win = game.get_training_frames(pname=pname)
        
        num_opponents = np.sum(opponent_features[0, :, 0] > 0)
        map_size = moves.shape[1]
        
        wk = str(num_opponents) + '_' + str(map_size) + '_' + str(default_lengths[map_size]) # weight key
        weights = WEIGHTS[wk]
        
        if WINDOW_SIZE is not None:
            if WINDOW_SIZE < 999:
                num_take = frames.shape[0]
            else:
                num_take = 4
            for i in range(num_take): # Take a few samples to relieve pre-compute cost
            
                if WINDOW_SIZE < 999:
                    end_ix = i + 1
                else:
                    end_ix = np.random.randint(frames.shape[0]) + 1
                
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
                
                _w_moves = moves[end_ix - 1] # Only relavent to the moves predicted

                w = weights[:, end_ix - 1]
                m_weights = w[_w_moves]
                
                # NEXT: add the weight matrix to the tuple (and use it during loss)
                
                
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

                tup = _frames, _moves, _generate, _my_player_features, _opponent_features, _will_have_ship, _should_construct, _did_win, m_weights

                queue.put(copy.deepcopy(tup))
        else:
            # Take all frames
            
            # Avoid GC issues
            _frames = copy.deepcopy(frames)
            _moves = copy.deepcopy(moves)
            _generate = copy.deepcopy(generate)
            _my_player_features = copy.deepcopy(my_player_features)
            _opponent_features = copy.deepcopy(opponent_features)
            _will_have_ship = copy.deepcopy(will_have_ship)
            _should_construct = copy.deepcopy(should_construct)
            _did_win = copy.deepcopy(did_win[0])

            shapes = [_frames.shape[0], _moves.shape[0], _generate.shape[0], _my_player_features.shape[0], _opponent_features.shape[0], _will_have_ship.shape[0], _should_construct.shape[0]]

            assert len(set(shapes)) == 1, print(shapes)
            
            tup = _frames, _moves, _generate, _my_player_features, _opponent_features, _will_have_ship, _should_construct, _did_win

            queue.put(copy.deepcopy(tup))

        del game # Should this be the last to be deleted?
        del frames
        del moves
        del generate
        del my_player_features
        del opponent_features
        del will_have_ship
        del should_construct
        del did_win
        del tup
        del _frames
        del _moves
        del _generate
        del _my_player_features
        del _opponent_features
        del _will_have_ship
        del _should_construct
        del _did_win

        if len(current) == 0:
            current = s_keep.copy()
            shuffle(current)
            #queue.put(None)
        

processes = []
for player in PLAYERS:

    # 5 queues, 1 for each map size (to improve compute efficiency)
    queues = [Queue(2) for _ in range(5)]
    queue_m_sizes = [32, 40, 48, 56, 64]
    #queue_m_sizes = [64, 64, 64, 64, 64]

    v_queues = [Queue(2) for _ in range(5)]
    v_queue_m_sizes = [32, 40, 48, 56, 64]
    #v_queue_m_sizes = [64, 64, 64, 64, 64]

    batch_queue = Queue(2)
    buffer_queue = PriorityQueue(max_buffer_size)

    v_batch_queue = Queue(2)
    v_buffer_queue = PriorityQueue(batch_size*3)

    processes += [Thread(target=worker, args=(queues[ix], queue_m_sizes[ix], player['pname'], player['train'])) for ix in range(5)]
    processes += [Thread(target=worker, args=(v_queues[ix], v_queue_m_sizes[ix], player['pname'], player['valid'])) for ix in range(5)]

    buffer_thread = Thread(target=buffer, args=(queues, buffer_queue, OVERLAP))
    batch_thread = Thread(target=batch_prep, args=(buffer_queue, batch_queue, MIN_BUFFER_SIZE))

    v_buffer_thread = Thread(target=buffer, args=(v_queues, v_buffer_queue, 1))
    v_batch_thread = Thread(target=batch_prep, args=(v_buffer_queue, v_batch_queue, batch_size*2))

    processes += [buffer_thread, batch_thread, v_buffer_thread, v_batch_thread]

    player['batch_q'] = batch_queue
    player['v_batch_q'] = v_batch_queue

[p.start() for p in processes]

model = Model(num_players=len(PLAYERS), window_size=WINDOW_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1) # Init lr doesn't matter. It gets set later.

best = np.array([999 for _ in range(len(PLAYERS))])
try:
    if RESTORE:
        print("Restoring...")
        model.load_state_dict(torch.load(os.path.join('/path/trained_models/', RESTORE_WHICH, "model.pt")))
        #model.eval()

    print("Training...")
    epochs_no_improvement = 0
    LR = 0.0009
    losses = []
    for step in range(20000000):
        player_batches = []
        for player in PLAYERS:
            batch = player['batch_q'].get()
            player_batches.append(batch)
        
        # TODO: For the case when either number of players is more than one
        # or batch size is more than one, I'll need to pad to max game length
        
        batch = [np.concatenate(x, 0) for x in zip(*player_batches)]
        shapes = [x.shape[0] for x in batch]
        assert len(set(shapes)) == 1
        f_batch, m_batch, g_batch, c_batch, t_batch, s_batch, h_batch, b_batch, w_batch, mw_batch = batch
#        print(s_batch.shape)
#        print(m_batch.shape)
#        print((m_batch[:, -1] == 5).sum())
#        print(((m_batch[:, -1] == 5) * s_batch).sum())
        #continue
        
        
        T = 400000
        M = 10
        t = step
        #S = 0.0003
        lr = (0.0006/2.)*(np.cos(np.pi*np.mod(t - 1, T/M)/(T/M)) + 1)
        
#        if step > 300000:
#            lr = 1e-5

        lr = LR

        for g in optimizer.param_groups:
            g['lr'] = lr
        
        loss = model(f_batch, c_batch, t_batch, my_ships=s_batch, moves=m_batch, generate=g_batch, train=True, will_have_ship=h_batch, should_construct=b_batch, did_win=w_batch, m_weights=mw_batch)

        #continue

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.cpu().data.numpy()
        losses.append(loss)
        epoch_size = int(76400/batch_size)
        if (step + 1) % epoch_size == 0 or step == 0:
            player_gen_losses = []
            player_average_frame_losses = []
            player_total_losses = []
            player_have_ship_losses = []
            player_should_construct_losses = []
            player_did_win_losses = []
            accuracies = []
            n_vsteps = int(58*400/batch_size)
            for vstep in range(n_vsteps): # Larger number will reduce variance; 400/batch_size steps will approximately validate 1 game for each map size for each player
            
                player_batches = []
                for player in PLAYERS:
                    batch = player['v_batch_q'].get()
                    player_batches.append(batch)
            
                batch = [np.concatenate(x, 0) for x in zip(*player_batches)]
                f_batch, m_batch, g_batch, c_batch, t_batch, s_batch, h_batch, b_batch, w_batch, mw_batch = batch
                
                #print((m_batch == 5).sum())
                
                with torch.no_grad():
#                    loss, gen_loss, frame_loss, hs_loss, total_loss, b_loss, w_loss, acc = model(f_batch, c_batch, t_batch, my_ships=s_batch, moves=m_batch, generate=g_batch, valid=True, will_have_ship=h_batch, should_construct=b_batch, did_win=w_batch, m_weights=mw_batch)
                    loss, gen_loss, frame_loss, total_loss, acc = model(f_batch, c_batch, t_batch, my_ships=s_batch, moves=m_batch, generate=g_batch, valid=True, will_have_ship=h_batch, should_construct=b_batch, did_win=w_batch, m_weights=mw_batch)
                
                loss = loss.cpu().data.numpy()
                gen_loss = gen_loss.cpu().data.numpy()
                frame_loss = frame_loss.cpu().data.numpy()
                #hs_loss = hs_loss.cpu().data.numpy()
                total_loss = total_loss.cpu().data.numpy()
                #b_loss = b_loss.cpu().data.numpy()
                #w_loss = w_loss.cpu().data.numpy()
                acc = acc.cpu().data.numpy()

                player_gen_losses.append(gen_loss)
                player_average_frame_losses.append(frame_loss)
                player_total_losses.append(total_loss)
                
                #player_have_ship_losses.append(hs_loss)
                #player_should_construct_losses.append(b_loss)
                #player_did_win_losses.append(w_loss)
                
                # TODO: accuracies are aggregated early on and assume 1 player.
                # Add support for multiple players.
                accuracies.append(acc)
                
                if step == 0 and vstep == 5:
                    break
        
            player_gen_losses = np.stack(player_gen_losses, 1)
            player_average_frame_losses = np.stack(player_average_frame_losses, 1)
            player_total_losses = np.stack(player_total_losses, 1)
            #player_have_ship_losses = np.stack(player_have_ship_losses, 1)
            #player_should_construct_losses = np.stack(player_should_construct_losses, 1)
            #player_did_win_losses = np.stack(player_did_win_losses, 1)
            player_accuracies = np.stack(accuracies)
            
            player_gen_losses = np.mean(player_gen_losses, 1)
            player_average_frame_losses = np.mean(player_average_frame_losses, 1)
            player_total_losses = np.mean(player_total_losses, 1)
            #player_have_ship_losses = np.mean(player_have_ship_losses, 1)
            #player_should_construct_losses = np.mean(player_should_construct_losses, 1)
            #player_did_win_losses = np.mean(player_did_win_losses, 1)
            
            vals = []
            for item in np.nanmean(player_accuracies, 0) * 100:
                if np.isnan(item):
                    vals.append('nan')
                else:
                    vals.append(f'{item:.1f}')
            
            #print('f{}'.format(*(np.nanmean(player_accuracies, 0) * 100)))

            
            assert player_total_losses.shape[0] == len(PLAYERS)
            
            player_print = " ".join(["{:.3f}/{:.3f}".format(x,y) for x,y in zip(player_average_frame_losses, player_gen_losses)])

            print_line = "{} T: {:.3f} V: ".format(step, np.mean(losses[-1000:])) + player_print + ' Acc: ' + ' '.join(vals)

            #current_loss = np.mean([x+y for x,y in zip(player_average_frame_losses, player_gen_losses)])

            # TODO: account for gen loss in here, too
            if np.sum(np.less(player_total_losses, best)) == len(PLAYERS): # All players must have improved
                best = player_total_losses
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_{}.pt'.format(step)))
                print(print_line + " *** new best ***")
                epochs_no_improvement = 0
            else:
                print(print_line)
                epochs_no_improvement += 1

            if epochs_no_improvement == 3:
                LR = LR/3
                epochs_no_improvement = 0
                print("New LR: {}".format(LR))

        # Overfit model to test architecture
#            for i in range(100000):
#                loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
#                print(loss)

except KeyboardInterrupt:
    print('Cleaning up')
    [p.join() for p in processes]
