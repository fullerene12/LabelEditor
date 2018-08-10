def find_bad_frame(data_set, check_progress=False, progress_bar=None):
    data_set_name = data_set.keys()[0][0]
    num_label = data_set[data_set_name].keys().levshape[0]
    data_set_size = data_set.shape[0]

    labels = list()
    for label_id in range(num_label):
        labels.append(data_set[data_set_name].keys()[label_id*3][0])

    bad_frames = list()
    for i in range(data_set_size):
        bad_frame = False

        for label in labels:
            if data_set[data_set_name][label]['likelihood'][i] < 0.95:
                bad_frame = True
            if data_set[data_set_name][label]['y'][i] > 350:
                bad_frame = True

        if bad_frame:
            bad_frames.append(i)

        if check_progress:
            if i%25 == 0:
                progress_bar.setValue(i*100.0/data_set_size)

    return bad_frames
