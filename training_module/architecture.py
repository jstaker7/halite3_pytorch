import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d as BN1d
from torch.nn import BatchNorm2d as BN2d
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DILATION = 1

SIZE = 32
L_SIZE = 32
F_SIZE = 16

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, bn=False):
        super(MLP, self).__init__()
    
        self.l1 = nn.Linear(input_size, output_size)
        
        if bn:
            self.bn = BN1d(output_size)
        else:
            self.bn = lambda x: x

    def forward(self, input):
    
        p = self.l1(input)
        a = F.relu(p)
        
        n = self.bn(a)

        return n

def pad_replay(frames, pad_amount=1):
    
    map_size = frames.shape[2]

    frames = torch.cat([frames[:, :, -pad_amount:], frames, frames[:, :, :pad_amount]], 2)
    frames = torch.cat([frames[:, :, :, -pad_amount:], frames, frames[:, :, :, :pad_amount]], 3)

    return frames


class Conv(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel, bn=True, downsample=False):
        super(Conv, self).__init__()

        self.l1 = nn.Conv2d(input_size, output_size, kernel, padding=(kernel-1)//2)
        self.l2 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        self.l3 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        self.l4 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        self.l5 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        self.l6 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        self.l7 = nn.Conv2d(output_size, output_size, kernel, padding=(kernel-1)//2)
        
        if downsample:
            self.downsample = Downsample(output_size, output_size, 3, True, 2)
        else:
            self.downsample = lambda x: x
        
        if bn:
            self.bn1 = BN2d(output_size)
            self.bn2 = BN2d(output_size)
            self.bn3 = BN2d(output_size)
            self.bn4 = BN2d(output_size)
            self.bn5 = BN2d(output_size)
            self.bn6 = BN2d(output_size)
            self.bn7 = BN2d(output_size)
        else:
            i = lambda x: x
            self.bn1 = i
            self.bn2 = i
            self.bn3 = i
            self.bn4 = i
            self.bn5 = i
            self.bn6 = i
            self.bn7 = i

    def forward(self, input):
#        print(input.shape)
#        input = pad_replay(input)
#        print(input.shape)
#        sdfsf

        p1 = self.l1(input)
        a1 = F.relu(p1)
        n1 = self.bn1(a1)
        
        p2 = self.l2(n1)
        a2 = F.relu(p2)
        n2 = self.bn2(a2)
        
        p3 = self.l3(n2) + n1
        a3 = F.relu(p3)
        n3 = self.bn3(a3)
        
        p4 = self.l4(n3)
        a4 = F.relu(p4)
        n4 = self.bn4(a4)
        
        p5 = self.l5(n4) + n3
        a5 = F.relu(p5)
        n5 = self.bn5(a5)
        
        p6 = self.l6(n5)
        a6 = F.relu(p6)
        n6 = self.bn6(a6)
        
#        p7 = self.l6(n6) #+ n5
#        a7 = F.relu(p7)
#        n7 = self.bn7(a7)

        o = self.downsample(n6)
        return o, p1

class Downsample(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel, bn=True, stride=2):
        super(Downsample, self).__init__()
    
        # TODO: calculate padding on the fly
        self.conv2d = nn.Conv2d(input_size, output_size, kernel, padding=1, stride=stride)
        self.pool = nn.MaxPool2d(2, stride=2)
        
        if bn:
            self.bn = BN2d(output_size)
        else:
            self.bn = lambda x: x

    def forward(self, input):
        p = self.conv2d(input)
        a = F.relu(p)
        n = self.bn(a)
        #p = self.pool(input) # hurt performance (even when added to n)
        o = n
        return o #, p

class Upsample(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel, bn=False, stride=2):
        super(Upsample, self).__init__()
    
        # TODO: calculate padding on the fly
        self.conv2dt = nn.ConvTranspose2d(input_size, output_size, kernel,
                                            stride=stride, padding=1,
                                            output_padding=1)
        
        if bn:
            self.bn = BN2d(output_size)
        else:
            self.bn = lambda x: x

    def forward(self, input, skip_con=None):
    
        p = self.conv2dt(input)
        
        if skip_con is not None:
            a = F.relu(p + skip_con)
        else:
            a = F.relu(p)

        return a

class ProcessMap(torch.nn.Module):
    def __init__(self, window_size):
        super(ProcessMap, self).__init__()
        
        map_input_size = 7*window_size + 6*(window_size-1)
        #map_input_size += F_SIZE*2 # include previous moves

        self.projection2d = Conv(map_input_size, SIZE, 1, bn=False)
        self.local_layers = nn.ModuleList([Conv(SIZE, SIZE, 3, downsample=True) for _ in range(7)])

    def forward(self, map):
        maps, pres = [], []
        map, _ = self.projection2d(map)
        maps.append(map)
        #pres.append(pre)
        
        for i in range(7):
            map, pre = self.local_layers[i](map)
            maps.append(map)
            pres.append(pre)
        return maps, pres

class Tower(torch.nn.Module):
    def __init__(self, num_players, window_size):
        super(Tower, self).__init__()
        
        self.process_map = ProcessMap(window_size)
        
        self.up_layers = nn.ModuleList([Upsample(SIZE, SIZE, 3, stride=2) for _ in range(6)] + [Upsample(L_SIZE, SIZE, 3, stride=2)])

        self.opponent_processor = MLP(3*window_size, F_SIZE)
        self.meta_processor = MLP(12*window_size, F_SIZE, bn=True)
        self.gamestate_processor = MLP(SIZE + F_SIZE*2, L_SIZE, bn=True)
        
        # TODO: unique "processors" for each player
    
        self.gen_predictor = nn.ModuleList([nn.Linear(L_SIZE, 1) for _ in range(num_players)])
        self.move_predictor = nn.ModuleList([nn.Conv2d(SIZE, 6, 1) for _ in range(num_players)])
    
        #self.should_l1s = nn.ModuleList([nn.Linear(L_SIZE, 32) for _ in range(num_players)])
        #self.will_have_l1s = nn.ModuleList([nn.Conv2d(SIZE, SIZE, 1) for _ in range(num_players)])
        #self.did_win_l1s = nn.ModuleList([nn.Linear(L_SIZE, L_SIZE) for _ in range(num_players)])
        #self.cell = torch.nn.LSTMCell(128, 128)

    def forward(self, frames, my_player_features, opponent_features,
                train=False, num_players=1,
                valid=False,
                prev_state=None):
        
        opps = self.opponent_processor(opponent_features)
        
        opp_sum = torch.sum(opps, 1)
        meta = self.meta_processor(my_player_features)
        
        
        maps, pres = self.process_map(frames)
 
        game_state = torch.cat([maps[-1].squeeze(-1).squeeze(-1), opp_sum, meta], 1)
        
        latent = self.gamestate_processor(game_state)
        
        maps[-1] = latent.unsqueeze(-1).unsqueeze(-1)
        
        #frames = frames.permute(0, 3, 1, 2)
        
#        s1 = ca.size()
#        s2 = tl.size()
#        s3 = frames.size()

        #expanded = torch.cat([ca.expand(s1[0], s1[1], s3[2], s3[3]), tl.expand(s2[0], s2[1], s3[2], s3[3]), frames], 1)
        
        #self.up_layers[-1]()
        
        map = maps[-1]
        #for i in reversed(range(1, 7)):
        for i in reversed(range(7)):
            pre = pres[i]
            map = self.up_layers[i](map, pre)

        player_generate_logits = []
        player_move_logits = []
        player_will_have_ship_logits = []
        player_should_construct_logits = []
        player_did_win_logits = []
        
        for i in range(num_players):
            generate_logits = self.gen_predictor[i](latent)

            moves_logits = self.move_predictor[i](map)

            player_generate_logits.append(generate_logits)
            player_move_logits.append(moves_logits)

        m_logits = torch.stack(player_move_logits)
        g_logits = torch.stack(player_generate_logits)
        
        m_probs = F.softmax(torch.stack(player_move_logits), dim=2)
        
        return m_logits, g_logits, m_probs, latent, player_move_logits, player_generate_logits
        

class Model(torch.nn.Module):
    def __init__(self, num_players, window_size):

        super(Model, self).__init__()
        
        self.tower = Tower(num_players, window_size)
    
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion2 = torch.nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, frames, my_player_features, opponent_features,
                train=False, num_players=1,
                learning_rate=None, my_ships=None, moves=None, generate=None,
                will_have_ship=None, should_construct=None, did_win=None,
                valid=False,
                init_state=None,
                m_weights=None):
        
        # These are labels and we don't use the history for now
        if generate is not None:
            generate = generate[:, -1:]
        if will_have_ship is not None:
            will_have_ship = will_have_ship[:, -1:]
        if should_construct is not None:
            should_construct = should_construct[:, -1:]
        
        frames = torch.tensor(frames, dtype=torch.float, device=device)
        my_player_features = torch.tensor(my_player_features, dtype=torch.float, device=device)
        opponent_features = torch.tensor(opponent_features, dtype=torch.float, device=device)
        
        if m_weights is not None:
            m_weights = torch.tensor(m_weights, dtype=torch.long, device=device)
        
        if moves is not None:
            moves = torch.tensor(moves, dtype=torch.long, device=device)
            moves = torch.unsqueeze(moves, -1)
            has_construct = moves == 5
            #moves = moves[:, -1] # Just takes last for now
        else:
            assert False # Assumed to have them for now
            
        moves, prev_moves = moves[:, -1:], moves[:, :-1]
        
        # Simply stack features for now (1 dim still kept to keep it compatible
        # with below loop)
        b, n, h, w, f = frames.size()
        frames = frames.permute(0, 1, 4, 2, 3).contiguous().view(b, 1, n*f, h, w)
        
        b, n, f = my_player_features.size()
        my_player_features = my_player_features.view(b, 1, n*f)
        
        b, n, o, f = opponent_features.size()
        opponent_features = opponent_features.permute(0, 2, 3, 1).contiguous().view(b, 1, o, n*f)

        if prev_moves.size()[1] != 0:
            pm_onehot = prev_moves.clone().repeat(1, 1, 1, 1, 6) # 6 moves
            pm_onehot.zero_()
            
            pm_onehot.scatter_(4, prev_moves, 1)
            
            pm_onehot = pm_onehot.permute(0, 1, 4, 2, 3)
            
            b, n, m, h, w = pm_onehot.size()
            pm_onehot = pm_onehot.contiguous().view(b, 1, n*m, h, w)
        
            frames = torch.cat([frames, pm_onehot.float()], 2)
        
        if train or valid:
            generate = torch.tensor(generate, dtype=torch.float, device=device)
            my_ships = torch.tensor(my_ships, dtype=torch.float, device=device)
            will_have_ship = torch.tensor(will_have_ship, dtype=torch.float, device=device)
            should_construct = torch.tensor(should_construct, dtype=torch.float, device=device)
            did_win = torch.tensor(did_win, dtype=torch.float, device=device)
        
            generate = torch.unsqueeze(generate, -1)
            
            my_ships = torch.unsqueeze(my_ships, 2)
            will_have_ship = torch.unsqueeze(will_have_ship, -1)
            should_construct = torch.unsqueeze(should_construct, -1)
            did_win = torch.unsqueeze(did_win, -1)
            
            # Because we are simply concating the features, we only need
            # the last mask
            my_ships = my_ships[:, -1:]
        
        if init_state is not None:
            prev_state = init_state
        else:
            # TODO: zero state
            prev_state = torch.zeros(1, 2)
            
        loss_history = []
        player_gen_losses_history = []
        player_average_frame_losses_history = []
        #player_have_ship_average_frame_losses_history = []
        player_total_losses_history = []
        #player_should_construct_losses_history = []
        #did_win_losses_history = []
        
        # NOTE: This will always be iter in loop; features are combined above.
        for i in range(frames.size()[1]):
        
#            m_logits, g_logits, m_probs, new_state, player_will_have_ship_logits, player_did_win_logits, player_should_construct_logits, player_move_logits, player_generate_logits = self.tower(frames[:, i], my_player_features[:, i], opponent_features[:, i],
#                    train=train, num_players=num_players,
#                    valid=valid,
#                    prev_state=prev_state)

            m_logits, g_logits, m_probs, new_state, player_move_logits, player_generate_logits = self.tower(frames[:, i], my_player_features[:, i], opponent_features[:, i],
                    train=train, num_players=num_players,
                    valid=valid,
                    prev_state=prev_state)

            prev_state = new_state
        
            if not train and not valid:
                m_logits = m_logits.cpu().data.numpy()
                g_logits = g_logits.cpu().data.numpy()
                m_probs = m_probs.cpu().data.numpy()
                new_state = new_state.cpu().data.numpy()
                return m_logits, g_logits, m_probs, new_state
            
            #h_logits = torch.sigmoid(torch.stack(player_will_have_ship_logits))
            #h_logits_raw = torch.stack(player_will_have_ship_logits)
            #b_logits = torch.stack(player_should_construct_logits)
            #w_logits = torch.stack(player_did_win_logits)

            cs = int(player_move_logits[0].size()[0] / num_players) # chunk size

            # TODO: Can be improved with gather_nd
            moves_logits = [torch.split(x, cs) for x in player_move_logits]
            generate_logits = [torch.split(x, cs) for x in player_generate_logits]
            #will_have_ship_logits = [torch.split(x, cs) for x in player_will_have_ship_logits]
            #should_construct_logits = [torch.split(x, cs) for x in player_should_construct_logits]
            #did_win_logits = [torch.split(x, cs) for x in player_did_win_logits]

            moves_logits = [x[i] for x, i in zip(moves_logits, range(num_players))]
            generate_logits = [x[i] for x, i in zip(generate_logits, range(num_players))]
            #will_have_ship_logits = [x[i] for x, i in zip(will_have_ship_logits, range(num_players))]
            #should_construct_logits = [x[i] for x, i in zip(should_construct_logits, range(num_players))]
            #did_win_logits = [x[i] for x, i in zip(did_win_logits, range(num_players))]

            moves_logits = torch.cat(moves_logits, 0)
            generate_logits = torch.cat(generate_logits, 0)

            #will_have_ship_logits = torch.cat(will_have_ship_logits, 0)
            #should_construct_logits = torch.cat(should_construct_logits, 0)
            #did_win_logits = torch.cat(did_win_logits, 0)

            frame_moves = torch.squeeze(moves[:, i], 3) # Too many dimensions
            
            #torch.set_printoptions(profile="full")
            

            losses = self.criterion(moves_logits, frame_moves)

            losses = losses * m_weights.float() # Weights for class balancing
            
            #frame_will_have_ship = will_have_ship[:, i].permute(0, 3, 1, 2)

            #have_ship_losses = self.criterion2(will_have_ship_logits, frame_will_have_ship)

            losses = torch.unsqueeze(losses, -1)
            
            frame_my_ships = my_ships[:, i]

#            with torch.no_grad():
#                have_ship_mask = torch.nn.functional.conv2d(frame_my_ships, self.kernel, padding=1)
#
#            have_ship_mask = have_ship_mask.sum(-1)
#
#            have_ship_mask = (have_ship_mask > 0.5).float()
#
#            have_ship_mask = torch.unsqueeze(have_ship_mask, -1)

            frame_my_ships = torch.squeeze(frame_my_ships, 1)
            frame_my_ships = torch.unsqueeze(frame_my_ships, -1)

            masked_loss = losses * frame_my_ships

            _, arged = torch.max(moves_logits, 1)

            frame_moves = frame_moves.unsqueeze(-1)

            arged = arged.unsqueeze(-1)

            is_correct = (arged == frame_moves).float()
            o_m = (frame_moves == 0).float() * frame_my_ships
            n_m = (frame_moves == 1).float() * frame_my_ships
            e_m = (frame_moves == 2).float() * frame_my_ships
            s_m = (frame_moves == 3).float() * frame_my_ships
            w_m = (frame_moves == 4).float() * frame_my_ships
            c_m = (frame_moves == 5).float() * frame_my_ships

            o_a = (o_m * is_correct).sum()/o_m.sum()
            n_a = (n_m * is_correct).sum()/n_m.sum()
            e_a = (e_m * is_correct).sum()/e_m.sum()
            s_a = (s_m * is_correct).sum()/s_m.sum()
            w_a = (w_m * is_correct).sum()/w_m.sum()
            c_a = (c_m * is_correct).sum()/c_m.sum()
            #c_a = (c_m * is_correct).sum()/torch.max(c_m.sum(), 1e-13*torch.ones_like(c_m[:, 0, 0, 0]))
            
            # Not storing history here; assuming 1 frame
            accuracies = torch.stack([o_a, n_a, e_a, s_a, w_a, c_a])

            #have_ship_losses = have_ship_losses * have_ship_mask

            ships_per_frame = frame_my_ships.sum(2).sum(1)

            #ship_positions_per_frame = have_ship_mask.sum(2).sum(1)

            frame_loss = masked_loss.sum(2).sum(1)

            #have_ship_frame_loss = have_ship_losses.sum(2).sum(1)

            average_frame_loss = frame_loss / torch.max(ships_per_frame, 1e-13*torch.ones_like(ships_per_frame)) # First frames have no ship

            #have_ship_average_frame_loss = have_ship_frame_loss / torch.max(ship_positions_per_frame, 1e-13*torch.ones_like(ship_positions_per_frame)) # First frames have no ship

            generate_losses = self.criterion2(generate_logits, generate[:, i])
            #should_construct_losses = self.criterion2(should_construct_logits, should_construct[:, i])
            #did_win_losses = self.criterion2(did_win_logits, did_win)

            # Individual losses for validation
            player_gen_losses = [x.mean() for x in torch.split(generate_losses, cs)]
            #player_should_construct_losses = [x.mean() for x in torch.split(should_construct_losses, cs)]
            #player_did_win_losses = [x.mean() for x in torch.split(did_win_losses, cs)]
#            print(average_frame_loss)
            player_average_frame_losses = [x.mean() for x in torch.split(average_frame_loss, cs)]

            #player_have_ship_average_frame_losses = [x.mean() for x in torch.split(have_ship_average_frame_loss, cs)]

            #player_total_losses = [x+0.02*y+0.00*z+0.0*w+0.00*k for x,y,z,w,k in zip(player_average_frame_losses, player_gen_losses, player_have_ship_average_frame_losses, player_should_construct_losses, player_did_win_losses)]
            player_total_losses = [x+0.02*y for x,y in zip(player_average_frame_losses, player_gen_losses)]

            generate_losses = generate_losses.mean()
            #should_construct_losses = should_construct_losses.mean()
            #did_win_losses = did_win_losses.mean()

            loss = average_frame_loss.mean() + 0.05 * generate_losses #+ 0.001 * have_ship_average_frame_loss.mean() + 0.0000000000001 * should_construct_losses + 0.000000000001 * did_win_losses

            player_gen_losses = torch.stack(player_gen_losses)
            player_average_frame_losses = torch.stack(player_average_frame_losses)
            #player_have_ship_average_frame_losses = torch.stack(player_have_ship_average_frame_losses)
            player_total_losses = torch.stack(player_total_losses)
            #player_should_construct_losses = torch.stack(player_should_construct_losses)
            #did_win_losses = torch.stack(player_did_win_losses)
            
            player_gen_losses_history.append(player_gen_losses)
            player_average_frame_losses_history.append(player_average_frame_losses)
            #player_have_ship_average_frame_losses_history.append(player_have_ship_average_frame_losses)
            player_total_losses_history.append(player_total_losses)
            #player_should_construct_losses_history.append(player_should_construct_losses)
            #did_win_losses_history.append(did_win_losses)
        
            loss_history.append(loss)
        
        loss = torch.mean(loss)
        
        player_gen_losses = torch.mean(torch.stack(player_gen_losses_history), 0)
        player_average_frame_losses = torch.mean(torch.stack(player_average_frame_losses_history), 0)
        #player_have_ship_average_frame_losses = torch.mean(torch.stack(player_have_ship_average_frame_losses_history), 0)
        player_total_losses = torch.mean(torch.stack(player_total_losses_history), 0)
        #player_should_construct_losses = torch.mean(torch.stack(player_should_construct_losses_history), 0)
        #did_win_losses = torch.mean(torch.stack(did_win_losses_history), 0)

        if train:
            return loss
        else:
            return loss, player_gen_losses, player_average_frame_losses, player_total_losses, accuracies




