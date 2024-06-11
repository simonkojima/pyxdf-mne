import logging
import numpy as np
import mne
import pyxdf

def get_name_streams(fname):
    streams, header = pyxdf.load_xdf(fname)

    return _get_name_streams(streams)

def _get_name_streams(streams):
    names = list()
    for stream in streams:
        names.append([stream['info']['name'][0], stream['info']['type'][0]])
    return names

def _get_marker_streams(streams):
    marker_streams = list()
    for stream in streams:
        if stream['info']['type'][0] == 'Markers':
            marker_streams.append(stream)
    if len(marker_streams) == 0:
        marker_streams = None
    return marker_streams

def get_channel_info(stream):
    ch_info = dict()
    ch_info['labels'] = list()
    ch_info['type'] = list()
    ch_info['unit'] = list()
    for ch in stream['info']['desc'][0]['channels'][0]['channel']:
        ch_info['labels'].append(ch['label'][0])
        ch_info['type'].append(ch['type'][0].lower())
        ch_info['unit'].append(ch['unit'][0])
    return ch_info

def read_raw_xdf(fname, ref_time, precise = False):
    """
    Parameters
    ==========
    fname: path-like
    ref_time: str
        Name of the stream which will be used for the time reference.
    precise: bool
        if True, timing synchronization between eeg and marker stream will be more precise. (slow)
    """

    print(fname)
    
    streams, header = pyxdf.load_xdf(fname)
    
    eeg = None
    marker = None

    for stream in streams:
        name = stream['info']['name'][0]
        if name == ref_time:
            eeg = stream
    marker = _get_marker_streams(streams)
    
    data = eeg['time_series'].T
    times = np.array(eeg['time_stamps'])
    
    fs = eeg['info']['nominal_srate'][0]
    n_channels = eeg['info']['channel_count'][0]
    ch_info = get_channel_info(eeg)
    
    raw = mne.io.RawArray(data = data,
                          info = mne.create_info(ch_names = ch_info['labels'], sfreq = fs, ch_types = ch_info['type']))
    

    if marker is not None:
        if len(marker) > 1:
            raise ValueError("multiple Marker streams are not supported currently.")

        events = marker[0]['time_series']
        mrk_times = np.array(marker[0]['time_stamps'])
        events = [int(val[0]) for val in events]

        if precise is False:
            annotations = mne.Annotations(mrk_times-times[0], 0, events)
            raw.set_annotations(annotations)

            return raw
        else:
            events_mne = list()
            for idx, event in enumerate(events):
                I = np.argmin(np.square(times - mrk_times[idx]))
                events_mne.append([I, 0, event])
            events = np.array(events_mne)
            
            return raw, events
            
    else:
        return raw




if __name__ == "__main__":
    import os
    home_dir = os.path.expanduser('~')
    files = os.listdir(home_dir)
    for file in files:
        if ".xdf" in file:
            break

    #names = get_name_streams(os.path.join(home_dir, file))
    #print(names)

    raw = read_raw_xdf(os.path.join(home_dir, file), ref_time = "BrainAmpSeries")
    print(raw)
    
    events, event_id = mne.events_from_annotations(raw)
    print(events)
    print(event_id)
    
