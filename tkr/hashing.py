from imports import *


def flatten(list_of_lists):
    "Flatten a list of lists into a single list."
    return [item for a_list in list_of_lists for item in a_list]


def make_constellation(spec, d_time, d_freq, threshold, rate=22050, dft_len=1024, step=512):
    '''Compute a constellation map.
    
    Args:
        spec (array): Spectrogram.
        d_time (float): Width of maximum-filter kernel.
        d_freq (float): Height of maximum-filter kernel.
        threshold (float): Threshold. Values below this threshold are thrown out.
        rate (int): Sampling rate. Measured in hertz.
        dft_len (int): Number of samples used in computing the DFT.
        step (int): Number of samples between adjacent blocks in the spectrogram.

    Returns:
        constellation (array): Constellation map with same dimensions as the input spectrogram.
    '''

    # Convert d_time and d_freq to samples.
    num_t = np.ceil(d_time * rate / step)
    num_f = np.ceil(d_freq * dft_len / rate)

    # Apply maximum filter; threshold. This makes the constellation map.
    filt_spec = nd.maximum_filter(spec, size=(num_t, num_f), mode='constant')
    constellation = np.logical_and(spec == filt_spec, filt_spec > threshold).astype(int)

    return constellation


def get_coordinates(constellation, order='(time, frequency)'):
    '''Given a constellation map, get the coordinates of non-zero values.
    
    Args:
        constellation (array): Constellation map.
        order (str): Choose (time, frequency) or (frequency, time) ordering for coordinates.

    Returns:
        coordinates (array): Array of coordinates of non-zero constellation values.
    '''

    where_peaks = np.array(np.where(constellation == 1))
    _, num_peaks = where_peaks.shape

    if order == '(time, frequency)':
        coordinates = np.array([(where_peaks[0][i], where_peaks[1][i]) for i in range(num_peaks)])
    elif order == '(frequency, time)':
        coordinates = np.array([(where_peaks[1][i], where_peaks[0][i]) for i in range(num_peaks)])
    else:
        raise Exception('Specify an ordering: (time, frequency) or (frequency, time).')

    return coordinates


def make_hash_list(items):    
    '''Make a hash list from items. 
    
    Args:
        items (list or array): Items from which a hash list is made.

    Returns:
        hash_list (dict): Hash list.
    '''

    if isinstance(items, np.ndarray):
        items = items.tolist()

    hash_list = {}
    for item in items:
        key, val = item
        if val in hash_list:
            hash_list[val].append(key)
        else:
            hash_list[val] = [key]

    hash_list = dict(sorted(hash_list.items())) # Sort
    return hash_list


def make_triplets(coordinates, dt=10, df=10):
    '''Convert a set of coordinates into a set of triplets.
    The coordinates are assumed to be in (time, frequency) order.
    
    Args:
        coordinates (array): Array of size (num_points, 2) with coordinates.
        dt (int): Consider times within dt samples of t.
        df (int): Consider frequencies within df samples of f.

    Returns:
        items (list): List of triplets.
    '''
    
    items = []
    for anchor in coordinates:
        t0, f0 = anchor
        for coordinate in coordinates:
            t1, f1 = coordinate
            if (t0 - dt < t1 < t0 + dt) and (f0 - df < f1 < f0 + df):
                items.append((t1, (f0, f1, t1 - t0)))
    return items




#############################
# PICK UP FROM HERE... FIX! #
#############################

def make_matching_function(hash_list, query, order='(time, frequency)'):
    # if order == '(time, frequency)':
    #     size = np.max(flatten(list(hash_list.values()))) - np.min(query[0, :])
    #     matching_function = np.zeros(size)
    # elif order == '(frequency, time)':
    #     size = np.max(flatten(list(hash_list.values()))) - np.min(query[1, :])
    #     matching_function = np.zeros(size)
    # else:
    #     raise Exception('Specify an ordering: (time, frequency) or (frequency, time).')
    matching_function = np.zeros(1000)
    
    for point in query:
        if order == '(time, frequency)':
            n, k = point
        elif order == '(frequency, time)':
            k, n = point
        else:
            raise Exception('Specify an ordering: (time, frequency) or (frequency, time).')
    
        if k in hash_list:
            for l in hash_list[k]:
                m = l - n
                matching_function[m] += 1

    return matching_function


def get_match(matching_function, tol=1e-9):
    match_score = np.max(matching_function)
    match_indices = np.where(np.abs(matching_function - match_score) < tol)[0]
    return match_indices, match_score
