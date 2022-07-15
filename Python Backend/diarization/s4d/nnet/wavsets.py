# -*- coding: utf-8 -*-
#
# This file is part of s4d.
#
# s4d is a python package for speaker diarization.
# Home page: http://www-lium.univ-lemans.fr/s4d/
#
# s4d is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# s4d is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with s4d.  If not, see <http://www.gnu.org/licenses/>.


"""
Copyright 2014-2021 Anthony Larcher
"""

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2021 Anthony Larcher and Sylvain Meignier"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

import numpy
import pathlib
import random
import scipy
import sidekit
import soundfile
import torch
import tqdm
import re
import yaml

from ..diar import Diar
from pathlib import Path
from sidekit.nnet.xsets import PreEmphasis
from sidekit.nnet.xsets import MFCC
from sidekit.nnet.xsets import CMVN
from sidekit.nnet.xsets import FrequencyMask
from sidekit.nnet.xsets import TemporalMask
from torch.utils.data import Dataset
from torchvision import transforms
from collections import namedtuple

#Segment = namedtuple('Segment', ['show', 'start_time', 'end_time'])

def read_ctm(filename, normalize_cluster=False, encoding="utf8"):
    """
    Read a segmentation file
    :param filename: the str input filename
    :return: a diarization object
    :param normalize_cluster: normalize the cluster by removing upper case
    """
    fic = open(filename, 'r', encoding=encoding)
    diarization = Diar()

    for line in fic:
        line = re.sub('\s+',' ',line)
        line = line.strip()
        if line.startswith('#') or line.startswith(';;'):
            continue
            # split line into fields
        show=line.split()[0]
        start= int(float(line.split()[2])*1000)/10.
        stop= start + int(float(line.split()[3])*1000)/10.
        word= line.split()[4]

    if normalize_cluster:
        word = str2str_normalize(word)

    diarization.append(show=show, cluster=word, start=float(start), stop=stop)
    fic.close()

    return diarization


def overlapping(seg1,seg2):
    seg1_start,seg1_stop=seg1
    seg2_start,seg2_stop=seg2
    if seg1_start <= seg2_start:
        return seg1_stop > seg2_start
    else:
        return seg2_stop > seg1_start


def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    return numpy.lib.stride_tricks.as_strided(sig,
                                           shape=shape,
                                           strides=strides).squeeze()

def load_wav_segment(wav_file_name, idx, duration, seg_shift, framerate=16000):
    """

    :param wav_file_name:
    :param idx:
    :param duration:
    :param seg_shift:
    :param framerate:
    :return:
    """
    # Load waveform
    signal = sidekit.frontend.io.read_audio(wav_file_name, framerate)[0]
    tmp = framing(signal,
                  int(framerate * duration),
                  win_shift=int(framerate * seg_shift),
                  context=(0, 0),
                  pad='zeros')
    return tmp[idx], len(signal)


def mdtm_to_label(mdtm_filename,
                  start_time,
                  stop_time,
                  sample_number,
                  speaker_dict,
                  is_uem=False,
                  is_ctm=False):
    """

    :param mdtm_filename:
    :param start_time:
    :param stop_time:
    :param sample_number:
    :param speaker_dict:
    :return:
    """
    if is_uem:
        diarization = Diar.read_uem(mdtm_filename)
    elif is_ctm:
        diarization = read_ctm(mdtm_filename)
    else:
        diarization = Diar.read_mdtm(mdtm_filename)
    diarization.sort(['show', 'start'])

    # When one segment starts just the frame after the previous one ends, o
    # we replace the time of the start by the time of the previous stop to avoid artificial holes
    previous_stop = 0
    for ii, seg in enumerate(diarization.segments):
        if ii == 0:
            previous_stop = seg['stop']
        else:
            if seg['start'] == diarization.segments[ii - 1]['stop'] + 1:
                diarization.segments[ii]['start'] = diarization.segments[ii - 1]['stop']

    # Create the empty labels
    label = []

    # Compute the time stamp of each sample
    time_stamps = numpy.zeros(sample_number, dtype=numpy.float32)
    period = (stop_time - start_time) / sample_number
    for t in range(sample_number):
        time_stamps[t] = start_time + (2 * t + 1) * period / 2

    for idx, time in enumerate(time_stamps):
        lbls = []
        for seg in diarization.segments:
            if seg['start'] / 100. <= time <= seg['stop'] / 100.:
                if speaker_dict is None:
                    lbls.append("1")
                else:
                    lbls.append(speaker_dict[seg['cluster']])

        if len(lbls) > 0:
            label.append(lbls)
        else:
            label.append([])

    if is_uem:
        tmp_label = []
        for lbl in label:
            tmp_label.append(len(lbl))
        label = tmp_label

    return label


def get_segment_label(label,
                      seg_idx,
                      mode,
                      duration,
                      framerate,
                      seg_shift,
                      collar_duration,
                      filter_type="gate"):
    """

    :param label:
    :param seg_idx:
    :param mode:
    :param duration:
    :param framerate:
    :param seg_shift:
    :param collar_duration:
    :param filter_type:
    :return:
    """

    # Create labels with Diracs at every speaker change detection
    spk_change = numpy.zeros(label.shape, dtype=int)
    spk_change[:-1] = label[:-1] ^ label[1:]
    spk_change = numpy.not_equal(spk_change, numpy.zeros(label.shape, dtype=int))

    # depending of the mode, generates the labels and select the segments
    if mode == "vad":
        output_label = (label > 0.5).astype(numpy.long)

    elif mode == "spk_turn":
        # Apply convolution to replace diracs by a chosen shape (gate or triangle)
        filter_sample = collar_duration * framerate * 2 + 1
        conv_filt = numpy.ones(filter_sample)
        if filter_type == "triangle":
            conv_filt = scipy.signal.triang(filter_sample)
        output_label = numpy.convolve(conv_filt, spk_change, mode='same')

    elif mode == "overlap":
        output_label = (label > 0.5).astype(numpy.long)

    else:
        raise ValueError("mode parameter must be 'vad', 'spk_turn' or 'overlap', 'resegmentation'")

    # Create segments with overlap
    segment_label = framing(output_label,
                  int(framerate * duration),
                  win_shift=int(framerate * seg_shift),
                  context=(0, 0),
                  pad='zeros')

    return segment_label[seg_idx]


def process_segment_label(label,
                          mode,
                          framerate,
                          collar_duration,
                          filter_type="gate"):
    """

    :param label:
    :param seg_idx:
    :param mode:
    :param duration:
    :param framerate:
    :param seg_shift:
    :param collar_duration:
    :param filter_type:
    :return:
    """

    # depending of the mode, generates the labels and select the segments
    if mode == "vad":
        output_label = numpy.array([len(a) > 0 for a in label]).astype(numpy.long)

    elif mode == "spk_turn":

        tmp_label = []
        for a in label:
            if len(a) == 0:
                tmp_label.append(0)
            elif len(a) == 1:
                tmp_label.append(a[0])
            else:
                tmp_label.append(sum(a) * 1000)

        label = numpy.array(tmp_label)

        # Create labels with Diracs at every speaker change detection
        spk_change = numpy.zeros(label.shape, dtype=int)
        spk_change[:-1] = label[:-1] ^ label[1:]
        spk_change = numpy.not_equal(spk_change, numpy.zeros(label.shape, dtype=int))

        # Apply convolution to replace diracs by a chosen shape (gate or triangle)
        filter_sample = int(collar_duration * framerate * 2 + 1)

        conv_filt = numpy.ones(filter_sample)
        if filter_type == "triangle":
            conv_filt = scipy.signal.triang(filter_sample)
        output_label = numpy.convolve(conv_filt, spk_change, mode='same')

    elif mode == "overlap":
        label = numpy.array([len(a) for a in label]).astype(numpy.long)

        # For the moment, we just consider two classes: overlap / no-overlap
        # in the future we might want to classify according to the number of speaker speaking at the same time
        output_label = (label > 1).astype(numpy.long)

    elif mode == "resegmentation":
        tmp_label = []
        for a in label:
            if len(a) == 0:
                tmp_label.append(0)
            elif len(a) == 1:
                tmp_label.append(a[0])
            else:
                tmp_label.append(sum(a) * 1000)

        output_label = numpy.array(tmp_label)

    else:
        raise ValueError("mode parameter must be 'vad', 'spk_turn' or 'overlap', 'resegmentation'")

    return output_label


def seqSplit(mdtm_dir,
             wav_dir,
             mode='vad',
             duration=2.,
             file_list=None):
    """
    
    :param mdtm_dir:
    :param mode: can be 'vad' or 'spk_turn'
    :param duration: 
    :return: 
    """
    segment_list = Diar()
    speaker_dict = dict()
    idx = 0

    # Get the list of shows to process
    if file_list is None:
        # For each MDTM
        show_list = []
        for mdtm_file in tqdm.tqdm(pathlib.Path(mdtm_dir).glob('*.mdtm')):
            show_list.append(str(mdtm_file)[len(mdtm_dir):].split(".")[0])
    else:
        with open(file_list, 'r') as fh:
            tmp = fh.readlines()
            show_list = [l.rstrip() for l in tmp if not l == '']

    #for mdtm_file in tqdm.tqdm(pathlib.Path(mdtm_dir).glob('*.mdtm')):
    for show_name in show_list:

        # Load MDTM file
        ref = Diar.read_mdtm(mdtm_dir + show_name + ".mdtm")
        ref.sort()
        last_stop = ref.segments[-1]["stop"]

        #showName = str(mdtm_file)[len(mdtm_dir):].split(".")[0]

        if mode == 'vad':
            _stop = float(ref.segments[-1]["stop"]) / 100.
            _start = float(ref.segments[0]["start"]) / 100.

            while _start + 2 * duration < _stop:
                segment_list.append(show=show_name,
                                    cluster="",
                                    start=_start,
                                    stop=_start + 2 * duration)
                _start += duration / 4.

        elif mode == "spk_turn" or mode == "overlap":
            # Get the borders of the segments (not the start of the first and not the end of the last

            # Check the length of audio
            nfo = soundfile.info(wav_dir + show_name + ".wav")

            # For each border time B get a segment between B - duration and B + duration
            # in which we will pick up randomly later
            for idx, seg in enumerate(ref.segments):

                if seg["start"] / 100. > duration and seg["start"] / 100. + duration < nfo.duration:
                    segment_list.append(show=seg['show'],
                                        cluster="",
                                        start=float(seg["start"]) / 100. - duration,
                                        stop=float(seg["start"]) / 100. + duration)
                if seg["stop"] / 100. > duration and seg["stop"] / 100. + duration < nfo.duration:
                    segment_list.append(show=seg['show'],
                                        cluster="",
                                        start=float(seg["stop"]) / 100. - duration,
                                        stop=float(seg["stop"]) / 100. + duration)

        # Get list of unique speakers
        speakers = ref.unique('cluster')
        for spk in speakers:
            if not spk in speaker_dict:
                speaker_dict[spk] =  idx
                idx += 1

    return segment_list, speaker_dict


class SeqSet(Dataset):
    """
    Object creates a dataset for sequence to sequence training
    """
    def __init__(self,
                 dataset_yaml,
                 wav_dir,
                 mdtm_dir,
                 mode,
                 segment_list=None,
                 speaker_dict=None,
                 filter_type="gate",
                 collar_duration=0.1,
                 audio_framerate=16000,
                 output_framerate=100,
                 output_sample_number=None):
        """

        :param wav_dir:
        :param mdtm_dir:
        :param mode:
        :param duration:
        :param filter_type:
        :param collar_duration:
        :param audio_framerate:
        :param output_framerate:
        :param transform_pipeline:
        """

        self.wav_dir = wav_dir
        self.mdtm_dir = mdtm_dir
        self.mode = mode
        self.filter_type = filter_type
        self.collar_duration = collar_duration
        self.audio_framerate = audio_framerate
        self.output_framerate = output_framerate
        self.output_sample_number = output_sample_number

        self.duration = dataset_yaml["train"]["duration"]

        if mode == "train":
            self.transformation  = dataset_yaml["train"]["transformation"]

        else:
            self.transformation  = dataset_yaml["eval"]["transformation"]


        self.segment_list = segment_list
        self.speaker_dict = speaker_dict
        self.len = len(segment_list)

        _transform = []
        if self.transformation["pipeline"] != '' and self.transformation["pipeline"] is not None:


            self.add_noise = numpy.zeros(self.len, dtype=bool)
            self.add_reverb = numpy.zeros(self.len, dtype=bool)
            self.spec_aug = numpy.zeros(self.len, dtype=bool)
            self.temp_aug = numpy.zeros(self.len, dtype=bool)

            trans = self.transformation["pipeline"].split(',')
            for t in trans:
                if 'PreEmphasis' in t:
                    _transform.append(PreEmphasis())

                if 'add_noise' in t:
                    self.add_noise[:int(self.len * self.transformation["noise_file_ratio"])] = 1
                    numpy.random.shuffle(self.add_noise)
                    _transform.append(AddNoise(noise_db_csv=self.transformation["noise_db_csv"],
                                               snr_min_max=self.transformation["noise_snr"],
                                               noise_root_path=self.transformation["noise_root_db"]))

                if 'add_reverb' in t:
                    has_pyroom = True
                    try:
                        import pyroomacoustics
                    except ImportError:
                        has_pyroom = False
                    if has_pyroom:
                        self.add_reverb[:int(self.len * self.transformation["reverb_file_ratio"])] = 1
                        numpy.random.shuffle(self.add_reverb)

                        _transform.append(AddReverb(depth=self.transformation["reverb_depth"],
                                                    width=self.transformation["reverb_width"],
                                                    height=self.transformation["reverb_height"],
                                                    absorption=self.transformation["reverb_absorption"],
                                                    noise=None,
                                                    snr=self.transformation["reverb_snr"]))

                if 'MFCC' in t:
                    _transform.append(MFCC())

                if "CMVN" in t:
                    _transform.append(CMVN())

                if "FrequencyMask" in t:
                    a = int(t.split('-')[0].split('(')[1])
                    b = int(t.split('-')[1].split(')')[0])
                    _transform.append(FrequencyMask(a, b))

                if "TemporalMask" in t:
                    a = int(t.split("(")[1].split(")")[0])
                    _transform.append(TemporalMask(a))

        self.transforms = transforms.Compose(_transform)

        if segment_list is None and speaker_dict is None:
            segment_list, speaker_dict = seqSplit(mdtm_dir=self.mdtm_dir,
                                                  duration=self.duration)
    def __getitem__(self, index):
        """
        On renvoie un segment wavform brut mais il faut que les labels soient échantillonés à la bonne fréquence
        (trames)
        :param index:
        :return:
        """
        # Get segment info to load from
        seg = self.segment_list[index]

        # Randomly pick an audio chunk within the current segment
        start = random.uniform(seg["start"], seg["start"] + self.duration)

        sig, _ = soundfile.read(self.wav_dir + seg["show"] + ".wav",
                                start=int(start * self.audio_framerate),
                                stop=int((start + self.duration) * self.audio_framerate)
                                )
        sig += 0.0001 * numpy.random.randn(sig.shape[0])

        if self.transformation:
            sig, speaker_idx, _, __, _t, _s = self.transforms((sig, None,  None, None, None, None))


        if self.output_sample_number is None:
            tmp_label = mdtm_to_label(mdtm_filename=self.mdtm_dir + seg["show"] + ".mdtm",
                                      start_time=start,
                                      stop_time=start + self.duration,
                                      sample_number=sig.shape[-1],
                                      speaker_dict=self.speaker_dict)

            label = process_segment_label(label=tmp_label,
                                          mode=self.mode,
                                          framerate=self.output_framerate,
                                          collar_duration=None)

        else:
            tmp_label = mdtm_to_label(mdtm_filename=self.mdtm_dir + seg["show"] + ".mdtm",
                                      start_time=start,
                                      stop_time=start + self.duration,
                                      sample_number=self.output_sample_number,
                                      speaker_dict=self.speaker_dict)

            label = process_segment_label(label=tmp_label,
                                          mode=self.mode,
                                          framerate=self.duration / float(self.output_sample_number),
                                          collar_duration=None)


        return torch.from_numpy(sig.T).type(torch.FloatTensor), torch.from_numpy(label.astype('long'))

    def __len__(self):
        return self.len


def create_train_val_seqtoseq(dataset_yaml):
    """

    :param self:
    :param wav_dir:
    :param mdtm_dir:
    :param mode:
    :param segment_list
    :param speaker_dict:
    :param filter_type:
    :param collar_duration:
    :param audio_framerate:
    :param output_framerate:
    :param transform_pipeline:
    :return:
    """
    with open(dataset_yaml, "r") as fh:
        dataset_params = yaml.load(fh, Loader=yaml.FullLoader)

    torch.manual_seed(dataset_params['seed'])

    # Read all MDTM files and ouptut a list of segments with minimum duration as well as a speaker dictionary
    segment_list, speaker_dict = seqSplit(mdtm_dir=dataset_params["mdtm_dir"],
                                          wav_dir=dataset_params["wav_dir"],
                                          duration=dataset_params["train"]["duration"],
                                          file_list=dataset_params["train"]["file_list"])

    split_idx = numpy.random.choice([True, False],
                                    size=(len(segment_list),),
                                    p=[1 - dataset_params["validation_ratio"], dataset_params["validation_ratio"]])
    segment_list_train = Diar.copy_structure(segment_list)
    segment_list_val = Diar.copy_structure(segment_list)
    for idx, seg in enumerate(segment_list.segments):
        if split_idx[idx]:
            segment_list_train.append_seg(seg)
        else:
            segment_list_val.append_seg(seg)

    # Split the list of segment between training and validation sets
    train_set = SeqSet(dataset_params,
                       wav_dir=dataset_params["wav_dir"],
                       mdtm_dir=dataset_params["mdtm_dir"],
                       mode=dataset_params["mode"],
                       segment_list=segment_list_train,
                       speaker_dict=speaker_dict,
                       filter_type=dataset_params["filter_type"],
                       collar_duration=dataset_params["collar_duration"],
                       audio_framerate=dataset_params["sample_rate"],
                       output_framerate=dataset_params["output_rate"],
                       output_sample_number=dataset_params["output_sample_number"])

    validation_set = SeqSet(dataset_params,
                            wav_dir=dataset_params["wav_dir"],
                            mdtm_dir=dataset_params["mdtm_dir"],
                            mode=dataset_params["mode"],
                            segment_list=segment_list_val,
                            speaker_dict=speaker_dict,
                            filter_type=dataset_params["filter_type"],
                            collar_duration=dataset_params["collar_duration"],
                            audio_framerate=dataset_params["sample_rate"],
                            output_framerate=dataset_params["output_rate"],
                            output_sample_number=dataset_params["output_sample_number"])

    return train_set, validation_set


def seqSplit_sliding_window(show,
                            mdtm_fn,
                            wav_fn,
                            duration=3.2,
                            speaker_dict=None,
                            shift=2.4):
    """

    :param mdtm_fn:
    :param wav_fn:
    :param uem_fn:
    :param duration:
    :param audio_framerate:
    :param shift:
    :return:
    """

    segment_list = Diar()

    ref = None
    if mdtm_fn is not None:
        if speaker_dict is None:
            speaker_dict = dict()

        # Load ref file
        ref = Diar.read_mdtm(mdtm_fn)

        # Get list of unique speakers
        idx = 0
        speakers = ref.unique('cluster')
        for spk in speakers:
            if not spk in speaker_dict:
                speaker_dict[spk] =  idx
                idx += 1

    # Check the length of audio
    nfo = soundfile.info(wav_fn)

    start = 0.
    while (start + duration < nfo.duration):
        segment_list.append(show=show,
                            cluster="",
                            start=start,
                            stop=start + duration)
        start += shift

    return segment_list, speaker_dict


class SeqSetSingle(Dataset):
    """
    Object creates a dataset for sequence to sequence training
    """
    def __init__(self,
                 show,
                 wav_fn,
                 mdtm_fn=None,
                 mode="vad",
                 audio_framerate=16000,
                 output_framerate=100,
                 speaker_dict=None,
                 duration=3.2,
                 shift=2.4,
                 transform_pipeline=""):
        """

        :param wav_dir:
        :param mdtm_dir:
        :param mode:
        :param duration:
        :param filter_type:
        :param collar_duration:
        :param audio_framerate:
        :param output_framerate:
        :param transform_pipeline:
        """

        self.wav_fn = wav_fn
        self.mdtm_fn = mdtm_fn
        self.mode = mode
        self.audio_framerate = audio_framerate
        self.output_framerate = output_framerate
        self.speaker_dict = speaker_dict
        self.duration = duration
        self.shift = shift
        self.transform_pipeline = transform_pipeline

        _transform = []
        if not self.transform_pipeline == '':
            trans = self.transform_pipeline.split(',')
            for t in trans:
                if 'PreEmphasis' in t:
                    _transform.append(PreEmphasis())
                if 'MFCC' in t:
                    _transform.append(MFCC())
                if "CMVN" in t:
                    _transform.append(CMVN())
                if "FrequencyMask" in t:
                    a = int(t.split('-')[0].split('(')[1])
                    b = int(t.split('-')[1].split(')')[0])
                    _transform.append(FrequencyMask(a, b))
                if "TemporalMask" in t:
                    a = int(t.split("(")[1].split(")")[0])
                    _transform.append(TemporalMask(a))

        self.transforms = transforms.Compose(_transform)

        segment_list, speaker_dict = seqSplit_sliding_window(show,
                                                             mdtm_fn=self.mdtm_fn,
                                                             wav_fn=self.wav_fn,
                                                             duration=self.duration,
                                                             speaker_dict=self.speaker_dict,
                                                             shift=self.shift)

        self.segment_list = segment_list
        self.speaker_dict = speaker_dict
        self.len = len(self.segment_list)

    def __getitem__(self, index):
        """
        On renvoie un segment wavform brut mais il faut que les labels soient échantillonés à la bonne fréquence
        (trames)
        :param index:
        :return:
        """
        # Get segment info to load from
        seg = self.segment_list[index]

        sig, _ = soundfile.read(self.wav_fn,
                                start=int(seg["start"] * self.audio_framerate),
                                stop=int((seg["start"] + self.duration) * self.audio_framerate))

        sig += 0.0001 * numpy.random.randn(sig.shape[0])

        if self.transform_pipeline:
            sig, speaker_idx, _, __, _t, _s = self.transforms((sig, None,  None, None, None, None))


        if self.speaker_dict is not None:
            tmp_label = mdtm_to_label(mdtm_filename=self.mdtm_fn,
                                      start_time=(seg["start"]/self.audio_framerate)*100,
                                      stop_time=(seg["stop"]/self.audio_framerate)*100,
                                      sample_number=sig.shape[1],
                                      speaker_dict=self.speaker_dict)

            label = process_segment_label(label=tmp_label,
                                          mode=self.mode,
                                          framerate=self.output_framerate,
                                          collar_duration=None,
                                          filter_type=None)

            return index, torch.from_numpy(sig.T).type(torch.FloatTensor), torch.from_numpy(label.astype('long'))

        else:
            return index, torch.from_numpy(sig.T).type(torch.FloatTensor), 0

    def __len__(self):
        return self.len
