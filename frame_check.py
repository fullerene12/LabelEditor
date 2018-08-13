def find_bad_frame(data_set, check_progress=False, progress_bar=None):
    data_set_name = data_set.keys()[0][0]
    num_label = data_set[data_set_name].keys().levshape[0]

    if check_progress:
        progress_bar.setValue(50)

    labels = list()
    for label_id in range(num_label):
        labels.append(data_set[data_set_name].keys()[label_id*3][0])

    label_bad_frame = data_set[data_set_name][labels[0]]['likelihood'] < 0.95
    for label in labels:
        label_bad_frame = ((data_set[data_set_name][label]['likelihood'] < 0.95) | label_bad_frame)

    bad_frames = data_set.index[label_bad_frame].tolist()
    return bad_frames
