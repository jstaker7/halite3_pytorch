import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d as BN1d
from torch.nn import BatchNorm2d as BN2d
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DILATION = 1

SIZE = 16
L_SIZE = 32
F_SIZE = 16

class Tower(torch.nn.Module):
    def __init__(self, num_players, window_size):
        super(Tower, self).__init__()

        self.conv1d_op1, self.bn_op1 = nn.Conv1d(3*window_size, F_SIZE, 1), BN1d(F_SIZE)
        #self.conv1d_op2, self.bn_op2 = nn.Conv1d(F_SIZE, F_SIZE, 1), BN1d(F_SIZE)
        
        self.dense_p1, self.bn_p1 = nn.Linear(12*window_size, F_SIZE), BN1d(F_SIZE)
        #self.dense_p2, self.bn_p2 = nn.Linear(F_SIZE, F_SIZE), BN1d(F_SIZE)
    
        #self.conv2d_1a, self.bn_1a = nn.Conv2d(7*window_size + 6*(window_size-1) + F_SIZE*2, SIZE, 1), BN2d(SIZE) # includes previous moves
        self.conv2d_1a, self.bn_1a = nn.Conv2d(7*window_size + 6*(window_size-1), SIZE, 1), BN2d(SIZE)
        #self.conv2d_1b, self.bn_1b = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_1d, self.bn_1d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_2a, self.bn_2a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_2d, self.bn_2d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_3a, self.bn_3a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_3d, self.bn_3d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_4a, self.bn_4a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_4d, self.bn_4d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_5a, self.bn_5a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_5d, self.bn_5d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_6a, self.bn_6a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2d_6d, self.bn_6d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        #self.conv2d_7a, self.bn_7a = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        #self.conv2d_7d, self.bn_7d = nn.Conv2d(SIZE, SIZE, 3, padding=1, stride=2), BN2d(SIZE)
    
        self.conv2d_l1, self.bn_l1 = nn.Conv2d(SIZE + F_SIZE*2, L_SIZE, 1), BN2d(L_SIZE)
    
        #self.conv2dt_1 = torch.nn.ConvTranspose2d(L_SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
        
        #self.conv2d_u1, self.bn_u1 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_2 = torch.nn.ConvTranspose2d(L_SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
    
        #self.conv2d_u2, self.bn_u2 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_3 = torch.nn.ConvTranspose2d(SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
        
        #self.conv2d_u3, self.bn_u3 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_4 = torch.nn.ConvTranspose2d(SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
    
        #self.conv2d_u4, self.bn_u4 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_5 = torch.nn.ConvTranspose2d(SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
    
        #self.conv2d_u5, self.bn_u5 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_6 = torch.nn.ConvTranspose2d(SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
    
        #self.conv2d_u6, self.bn_u6 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        self.conv2dt_7 = torch.nn.ConvTranspose2d(SIZE, SIZE, 3, stride=2, padding=1, output_padding=1)
    
        #self.conv2d_f1, self.bn_f1 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        #self.conv2d_f2, self.bn_f2 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        #self.conv2d_f3, self.bn_f3 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
        #self.conv2d_f4, self.bn_f4 = nn.Conv2d(SIZE, SIZE, 3, padding=1), BN2d(SIZE)
    
        #self.gen_l1s = nn.ModuleList([nn.Linear(L_SIZE, SIZE) for _ in range(num_players)])
        #self.gen_bn1s = nn.ModuleList([BN1d(SIZE) for _ in range(num_players)])
    
        #self.gen_l2s = nn.ModuleList([nn.Linear(SIZE, 32) for _ in range(num_players)])
        #self.gen_bn2s = nn.ModuleList([BN1d(32) for _ in range(num_players)])
    
        self.gen_l3s = nn.ModuleList([nn.Linear(L_SIZE, 1) for _ in range(num_players)])
    
        #self.move_l1s = nn.ModuleList([nn.Conv2d(SIZE, SIZE, 1) for _ in range(num_players)])
        #self.move_bn1s = nn.ModuleList([BN2d(SIZE) for _ in range(num_players)])
    
        self.move_l2s = nn.ModuleList([nn.Conv2d(SIZE, 6, 1) for _ in range(num_players)])
    
        #self.should_l1s = nn.ModuleList([nn.Linear(L_SIZE, 32) for _ in range(num_players)])
        #self.should_l2s = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_players)])
    
        #self.will_have_l1s = nn.ModuleList([nn.Conv2d(SIZE, SIZE, 1) for _ in range(num_players)])
        #self.will_have_l2s = nn.ModuleList([nn.Conv2d(SIZE, 1, 1) for _ in range(num_players)])
    
        #self.did_win_l1s = nn.ModuleList([nn.Linear(L_SIZE, L_SIZE) for _ in range(num_players)])
        #self.did_win_l2s = nn.ModuleList([nn.Linear(L_SIZE, L_SIZE) for _ in range(num_players)])
        #self.did_win_l3s = nn.ModuleList([nn.Linear(L_SIZE, 1) for _ in range(num_players)])
        
        #self.cell = torch.nn.LSTMCell(128, 128)

    def forward(self, frames, my_player_features, opponent_features,
                train=False, num_players=1,
                valid=False,
                prev_state=None):
        
#        shape = frames.size()
#
#        batch_size = shape[0]
#        num_frames = shape[1]
#
#        frames = frames.view(batch_size * num_frames, *[x for x in shape[2:]])
#        my_player_features = my_player_features.view(batch_size * num_frames, *[x for x in shape[2:]])
#        opponent_features = opponent_features.view(batch_size * num_frames, *[x for x in shape[2:]])

        
 #       if train or valid:
            
#            generate = generate.view(batch_size * num_frames, *[x for x in shape[2:]])
#            moves = moves.view(batch_size * num_frames, *[x for x in shape[2:]])
#            my_ships = my_ships.view(batch_size * num_frames, *[x for x in shape[2:]])
#            will_have_ship = will_have_ship.view(batch_size * num_frames, *[x for x in shape[2:]])
#            should_construct = should_construct.view(batch_size * num_frames, *[x for x in shape[2:]])
#            did_win = did_win.view(batch_size * num_frames, *[x for x in shape[2:]])

        opponent_features = opponent_features.permute(0, 2, 1)
        
        #ca = self.bn_op1(F.relu(self.conv1d_op1(opponent_features)))
        ca = self.bn_op1(torch.sum(F.relu(self.conv1d_op1(opponent_features)), 2)) # TODO: Sum before the relu?
        #ca = self.bn_op2(torch.sum(F.relu(self.conv1d_op2(ca)), 2))
        ca = ca.unsqueeze(-1).unsqueeze(-1)

        tl = self.bn_p1(F.relu(self.dense_p1(my_player_features)))
        #tl = self.bn_p2(F.relu(self.dense_p2(tl)))
        tl = tl.unsqueeze(-1).unsqueeze(-1)
        
        #frames = frames.permute(0, 3, 1, 2)
        
        s1 = ca.size()
        s2 = tl.size()
        s3 = frames.size()
        
        #expanded = torch.cat([ca.expand(s1[0], s1[1], s3[2], s3[3]), tl.expand(s2[0], s2[1], s3[2], s3[3]), frames], 1)
        
        d_l2_a_1_pre = self.conv2d_1a(frames) # 128
        #d_l2_a_1_pre = self.conv2d_1a(expanded) # 128
        
        d_l2_a_1 = self.bn_1a(F.relu(d_l2_a_1_pre))
        #_d_l2_a_1 = self.bn_1b(F.relu(self.conv2d_1b(d_l2_a_1)))
        d_l2_p = self.bn_1d(F.relu(self.conv2d_1d(d_l2_a_1))) # 64

        #d_l3_a_pre = self.conv2d_2a(d_l2_p)
        #d_l3_a = self.bn_2a(F.relu(d_l3_a_pre))
        #d_l3_a = d_l3_a_pre
        d_l3_a_pre = d_l2_p
        d_l3_a = d_l2_p
        d_l3_p = self.bn_2d(F.relu(self.conv2d_2d(d_l3_a))) # 32

        d_l4_a_pre = d_l3_p#self.conv2d_3a(d_l3_p)
        d_l4_a = d_l3_p#self.bn_3a(F.relu(d_l4_a_pre))
        #d_l4_a = d_l4_a_pre
        d_l4_p = self.bn_3d(F.relu(self.conv2d_3d(d_l4_a))) # 16

        d_l5_a_pre = d_l4_p#self.conv2d_4a(d_l4_p)
        d_l5_a = d_l4_p#self.bn_4a(F.relu(d_l5_a_pre))
        #d_l5_a = d_l5_a_pre
        d_l5_p = self.bn_4d(F.relu(self.conv2d_4d(d_l5_a))) # 8

        d_l6_a_pre = d_l5_p#self.conv2d_5a(d_l5_p)
        d_l6_a = d_l5_p#self.bn_5a(F.relu(d_l6_a_pre))
        #d_l6_a = d_l6_a_pre
        d_l6_p = self.bn_5d(F.relu(self.conv2d_5d(d_l6_a))) # 4

        d_l7_a_pre = d_l6_p#self.conv2d_6a(d_l6_p)
        d_l7_a = d_l6_p#self.bn_6a(F.relu(d_l7_a_pre))
        #d_l7_a = d_l7_a_pre
        d_l7_p = self.bn_6d(F.relu(self.conv2d_6d(d_l7_a))) # 2

#        d_l8_a_2_pre = self.conv2d_7a(d_l7_p)
#        #d_l8_a_2 = self.bn_7a(F.relu(d_l8_a_2_pre))
#        d_l8_a_2 = d_l8_a_2_pre
#        d_l8_p = self.bn_7d(F.relu(self.conv2d_7d(d_l8_a_2))) # 1

        final_state = torch.cat([d_l7_p, ca, tl], 1)
        latent = self.bn_l1(F.relu(self.conv2d_l1(final_state)))
        
        #latent = self.cell(latent, prev_state)
        
        #u_l8_a = self.conv2dt_1(latent) # 2

        #u_l8_c = F.relu(u_l8_a + d_l8_a_2_pre)

        #u_l8_s = self.bn_u1(F.relu(self.conv2d_u1(u_l8_c)))

        u_l7_a = self.conv2dt_2(latent) # 4

        u_l7_c = F.relu(u_l7_a + d_l7_a_pre)
        u_l7_s = u_l7_c#self.bn_u2(F.relu(self.conv2d_u2(u_l7_c)))

        u_l6_a = self.conv2dt_3(u_l7_s) # 8
        u_l6_c = F.relu(u_l6_a + d_l6_a_pre)
        u_l6_s = u_l6_c#self.bn_u3(F.relu(self.conv2d_u3(u_l6_c)))

        u_l5_a = self.conv2dt_4(u_l6_s) # 16
        u_l5_c = F.relu(u_l5_a + d_l5_a_pre)
        u_l5_s = u_l5_c#self.bn_u4(F.relu(self.conv2d_u4(u_l5_c)))

        u_l4_a = self.conv2dt_5(u_l5_s) # 32
        u_l4_c = F.relu(u_l4_a + d_l4_a_pre)
        u_l4_s = u_l4_c#self.bn_u5(F.relu(self.conv2d_u5(u_l4_c)))

        u_l3_a = self.conv2dt_6(u_l4_s) # 64
        u_l3_c = F.relu(u_l3_a + d_l3_a_pre)
        u_l3_s = u_l3_c#self.bn_u6(F.relu(self.conv2d_u6(u_l3_c)))

        u_l2_a = self.conv2dt_7(u_l3_s) # 128
        u_l2_c = F.relu(u_l2_a + d_l2_a_1_pre)
        
        u_l2_s_2 = u_l2_c#self.bn_f1(F.relu(self.conv2d_f1(u_l2_c)))
        #u_l2_s_2 = self.bn_f2(F.relu(self.conv2d_f2(u_l2_s_2)))
        #u_l2_s_2 = self.bn_f3(F.relu(self.conv2d_f3(u_l2_s_2)))
        #u_l2_s_2 = self.bn_f4(F.relu(self.conv2d_f4(u_l2_s_2)))

        player_generate_logits = []
        player_move_logits = []
        player_will_have_ship_logits = []
        player_should_construct_logits = []
        player_did_win_logits = []
        
        latent = torch.squeeze(latent, 2)
        latent = torch.squeeze(latent, 2)
        
        # TODO: more layers here that can be removed
        for i in range(num_players):
            #gen_latent1 = F.relu(self.gen_l1s[i](latent))
            #gen_latent1 = self.gen_bn1s[i](gen_latent1)
            #gen_latent = F.relu(self.gen_l2s[i](gen_latent1))
            #gen_latent = self.gen_bn2s[i](gen_latent)
            #generate_logits = self.gen_l3s[i](gen_latent1)
            
            generate_logits = self.gen_l3s[i](latent)

#            moves_latent = F.relu(self.move_l1s[i](u_l2_s_2))
#            moves_latent = self.move_bn1s[i](moves_latent)
#            moves_logits = self.move_l2s[i](moves_latent)

            moves_logits = self.move_l2s[i](u_l2_s_2)

#            if train or valid:
#                should_construct_latent = F.relu(self.should_l1s[i](latent))
#                should_construct_logits = self.should_l2s[i](should_construct_latent)
#
#                will_have_ship_latent = F.relu(self.will_have_l1s[i](u_l2_s_2))
#                will_have_ship_logits = self.will_have_l2s[i](will_have_ship_latent)
#
#                did_win_latent1 = F.relu(self.did_win_l1s[i](latent))
#                did_win_latent = F.relu(self.did_win_l2s[i](did_win_latent1))
#                did_win_logits = self.did_win_l3s[i](did_win_latent)
#
#                player_will_have_ship_logits.append(will_have_ship_logits)
#                player_should_construct_logits.append(should_construct_logits)
#                player_did_win_logits.append(did_win_logits)

            player_generate_logits.append(generate_logits)
            player_move_logits.append(moves_logits)

        m_logits = torch.stack(player_move_logits)
        g_logits = torch.stack(player_generate_logits)
        
        #print(torch.stack(player_move_logits).size())
        m_probs = F.softmax(torch.stack(player_move_logits), dim=2)
        
        return m_logits, g_logits, m_probs, latent, player_move_logits, player_generate_logits
        

class Model(torch.nn.Module):
    def __init__(self, num_players, window_size):

        super(Model, self).__init__()
        
        self.tower = Tower(num_players, window_size)
    
        kernel = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                 [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 1], [0, 0, 0]]]

        kernel = np.transpose(kernel, (0, 1, 2))

        kernel = np.expand_dims(kernel, 1)
        
        self.kernel = torch.from_numpy(kernel).float().to(device)
    
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
        
        #print((moves == 5).float().sum())
        
        #return
        
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
            
            #print(c_m.sum())

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
#            print(player_average_frame_losses)
#            print(average_frame_loss.size())
#            print(torch.stack(player_average_frame_losses))
#            dsfsf
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




