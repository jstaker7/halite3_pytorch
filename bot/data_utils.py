import numpy as np

def center_frames(frames, moves=None, include_shift=False):
    my_factory = frames[0, :, :, 3] > 0.5
    pos = np.squeeze(np.where(my_factory>0))
    expected_pos = np.squeeze(my_factory.shape)//2 # Assuming always square
    shift = expected_pos - pos
    frames = np.roll(frames, shift[0], axis=1)
    frames = np.roll(frames, shift[1], axis=2)
    
    if moves is not None and not include_shift:
        moves = np.roll(moves, shift[0], axis=2)
        moves = np.roll(moves, shift[1], axis=3)
        return frames, moves
    
    elif moves is not None and include_shift:
        moves = np.roll(moves, shift[0], axis=2)
        moves = np.roll(moves, shift[1], axis=3)
        return frames, moves, shift
    
    if include_shift:
        return frames, shift
    else:
        return frames

def pad_replay(frames, moves=None):
    
    map_size = frames.shape[1]

    total_x_left_padding = 0
    total_x_right_padding = 0
    total_y_left_padding = 0
    total_y_right_padding = 0

    # Ensure all get padded to the same max dim
    #max_dim = 192
    max_dim = 64 # For easier u-net implementation
    while frames.shape[1] != max_dim:
        pad_y1 = (min(max_dim, frames.shape[1]*3) - frames.shape[1])//2
        pad_y2 = (min(max_dim, frames.shape[1]*3) - frames.shape[1]) - pad_y1
        frames = np.concatenate([frames[:, -pad_y1:], frames, frames[:, :pad_y2]], axis=1)
        
        total_y_left_padding += pad_y1
        total_y_right_padding += pad_y2
        
        if moves is not None:
            moves = np.concatenate([moves[:, :, -pad_y1:], moves, moves[:, :, :pad_y2]], axis=2)
        
        pad_x1 = (min(max_dim, frames.shape[2]*3) - frames.shape[2])//2
        pad_x2 = (min(max_dim, frames.shape[2]*3) - frames.shape[2]) - pad_x1
        frames = np.concatenate([frames[:, :, -pad_x1:], frames, frames[:, :, :pad_x2]], axis=2)
        
        total_x_left_padding += pad_x1
        total_x_right_padding += pad_x2
    
        if moves is not None:
            moves = np.concatenate([moves[:, :, :, -pad_x1:], moves, moves[:, :, :, :pad_x2]], axis=3)
    
    padding = total_x_left_padding, total_x_right_padding, total_y_left_padding, total_y_right_padding
    
    if moves is not None:
        return frames, moves.astype('uint8'), padding

    return frames, padding

