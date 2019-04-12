import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import errno
import os
import csv
import shutil
import tensorflow as tf


def binary_array_to_int(array, size):
    out_list = []
    for i in range(size):
        input = array[i]
        out = np.argmax(input)
        out_list.append(out)
    out_arr = np.array(out_list)
    return out_arr


def simple_binary_array_to_int(array):
    input = array
    out = 0
    for bit in input:
        out = (out << 1) | bit
    return out


def my_print(content, signal):
    print(signal*10, content, signal*10)


def write_file(path, content, overwrite=False):
    """ write data to file.
    :param path:
    :param content:
    :param overwrite: open file by 'w' (True) or 'a' (False)
    :return:
    """
    # Check the file path.
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # Write data.
    if overwrite is True:
        with open(path, 'w') as fo:
            fo.write(str(content))
            fo.close()
    else:
        with open(path, 'a') as fo:
            fo.write(str(content) + '\n')
            fo.close()


def read_file(path):
    # Check the file path.
    if os.path.exists(path) is True:
        with open(path, 'r') as fo:
            data = fo.read()
            fo.close()
    else:
        data = None
    return data


def write_ndarray(path, data):
    """ write ndarray to csv file.
    """
    # Write data.
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        if os.path.exists(path) is True:
            old_array = np.loadtxt(path)
            new_array = np.concatenate((old_array, data))
            np.savetxt(path, new_array)
        else:
            np.savetxt(path, data)


def read_ndarray(path):
    """ read ndarray from file.
    """
    if os.path.exists(path) is True:
        data = np.loadtxt(path)
    else:
        data = None
    return data


def copy_rename_folder(oldpath, newpath, new_name):
    """ copy a folder and rename it.
    :param oldpath: string
    :param newpath: string
    :param new_name: string
    :return:
    """
    # Check the old path.
    if os.path.exists(os.path.dirname(oldpath)):
        try:
            shutil.copytree(oldpath, newpath)
            os.rename(newpath, new_name)
        except OSError as exc:  # Guard against race condition
            print('Warning: old path is not valid!')
            if exc.errno != errno.EEXIST:
                raise


def restore_parameters(sess, restore_path):
    """ Save and restore Network's weights.
    """
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint = tf.train.get_checkpoint_state(restore_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def save_parameters(sess, save_path, saver, name):
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    saver.save(sess, name)
    my_print('save weights', '-')


def read_output_plot(path, savepath, if_close_figure):
    interval = 100
    sum_size = 60000

    with open(path, 'r') as f:
        data_str = f.read()
        data_list = data_str.split('\n')
        size = len(data_list) - 1
        for i in range(size):
            data_list[i] = float(data_list[i].split()[1])

        # assert str not in [type(i) for i in data_list]
        if type(data_list[-1]) is not float:
            data_list.pop()  # data_list[-1] is '', so we need to pop it out.
        data = np.array(data_list)[:sum_size].astype(int)

        # data = data.copy()[30000:60000]

        data_plot = []
        size = int(len(data)/interval)
        for i in range(size):
            start = i*interval
            end = (i+1)*interval
            segment = data[start:end]
            data_plot.append(np.mean(segment))
        x_axis_data = np.arange(0, sum_size, interval)

        # x_axis_data = np.arange(30000, 60000, interval)

        write_file(savepath + 'data.txt', data_plot, True)
        write_file(savepath + 'data.txt', max(data_plot), False)

        plt.plot(x_axis_data, np.asarray(data_plot), label=path.split('/')[-2])
        plt.title(path)
        plt.xlabel('episodes')
        plt.ylabel('episode rewards')
        # x_axis_ticks = [0, 10000, 20000, 30000, 40000, 50000]
        # y_axis_ticks = [0, 10, 20, 30, 40, 50, 60]
        # plt.yticks(y_axis_ticks)
        # for items in x_axis_ticks:
        #     plt.vlines(items, min(y_axis_ticks), max(y_axis_ticks), colors="#D3D3D3", linestyles="dashed")
        # for items in y_axis_ticks:
        #     plt.hlines(items, x_axis_data.min(), x_axis_data.max(), colors="#D3D3D3", linestyles="dashed")
        plt.legend(loc='best')
        plt.savefig(savepath + 'data.png')
        if if_close_figure is True:
            plt.close()  # if not close figure, then all plot will be drawn in the same figure.
        # plt.show()


def main():
    dqn_list = [4]
    for ind in dqn_list:
        read_output_plot('../backup/data/' + str(ind) + '/data', '../backup/data/' + str(ind) + '/', False)
    # read_output_plot('../logs/11/data', '../logs/11/data.png')


if __name__ == '__main__':
    main()
