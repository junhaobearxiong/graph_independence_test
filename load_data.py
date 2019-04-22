import networkx as nx
import numpy as np

dwi_dirname = 'data/mri/dwi/'
dwi_file_format = 'sub-{}_ses-{}_dwi_desikan.ssv'
fmri_dirname = 'data/mri/fmri/'
fmri_file_format = 'sub-{}_ses-{}_bold_desikan_res-2x2x2_measure-correlation.gpickle'


def read_dwi(subject_id, session_num):
    file_name = dwi_dirname+dwi_file_format.format(subject_id, session_num)
    nx_out = nx.read_weighted_edgelist(file_name)
    return nx.to_numpy_array(nx_out)


def read_fmri(subject_id, session_num):
    file_name = fmri_dirname+fmri_file_format.format(subject_id, session_num)
    nx_out = nx.read_gpickle(file2)
    return nx.to_numpy_array(nx_out)
