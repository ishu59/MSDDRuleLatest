from Trie.MSDDNode import MSDDToken


def create_msdd_token_1d(stream_list, start_time):
    msdd_list = []
    for i, s in enumerate(stream_list):
        msdd_list.append(MSDDToken(value=s, stream_index=0, time_offset=start_time + i))
    return msdd_list


def create_msdd_token(stream_list, start_time):
    msdd_list = []
    if all(isinstance(e, (list, tuple)) for e in stream_list):
        for i, s in enumerate(stream_list):
            for j, t in enumerate(s):
                msdd_list.append(MSDDToken(value=t, stream_index=j, time_offset=start_time + i))
        return msdd_list
    else:
        for i, s in enumerate(stream_list):
            msdd_list.append(MSDDToken(value=s, stream_index=0, time_offset=start_time + i))
        return msdd_list
