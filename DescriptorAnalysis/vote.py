def check_global_id(global_id_dict):
    total_size = 0
    max_size_global_id = -1
    max_size = -1
    for global_id, count in global_id_dict.items():
        if global_id == 'unknown':
            continue
        total_size += count
        if count > max_size:
            max_size = count
            max_size_global_id = global_id
    if total_size > 0 and max_size / total_size > 0.5:
        return max_size_global_id, max_size
    else:
        return 'unknown_50', -1

def get_global_id_of_instance(merge_instance):
    init_num = 0
    for index, substring in enumerate(merge_instance):
        if substring=='#':
            init_num += 1
            if init_num == 2:
                start = index +1
    global_id = merge_instance[start:]
    return global_id