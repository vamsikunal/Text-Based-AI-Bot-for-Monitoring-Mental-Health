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

import logging
import numpy
import shutil
import torch
import tqdm
import yaml

from collections import OrderedDict
from sidekit.nnet.sincnet import SincNet
from torch.utils.data import DataLoader

from .wavsets import SeqSet
from .wavsets import create_train_val_seqtoseq

__license__ = "LGPL"
__author__ = "Anthony Larcher, Martin Lebourdais, Meysam Shamsi"
__copyright__ = "Copyright 2015-2021 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reS'


logging.basicConfig(format='%(asctime)s %(message)s')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """
    :param state:
    :param is_best:
    :param filename:
    :param best_filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def init_weights(m):
    """

    :return:
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _unfold(final_output, output, shift, output_rate):
    """
    append the output of one batch sample with overlapp to the final output sequence without overlap
    The overlap part will be gotten sumation
    """

    # Case where the output is the target and not data
    if len(output.shape) < 2:
        output = numpy.array([output])

    len_seq = output.shape[1]
    sampleNumber = int(shift * output_rate)

    for i in range(len(output)):
        if len(final_output) == 0:
            final_output = numpy.array(output[i]).tolist()
        else:
            final_output = final_output[:-len_seq + sampleNumber] \
                           + (numpy.array(final_output[-len_seq + sampleNumber:]) + numpy.array(output[i])[
                                                                                    :-sampleNumber]).tolist() \
                           + numpy.array(output[i]).tolist()[-sampleNumber:]

    return final_output

def multi_label_combination(output_idx, output_target, output_data, shift, output_rate, mode="mean"):
    """

    :param output_idx:
    :param output_target:
    :param output_data:
    :param shift:
    :param output_rate:
    :param mode:
    :return:
    """
    win_shift = int(shift * output_rate)

    # Initialize the size of final_output
    final_output = numpy.zeros((win_shift * (len(output_data) - 1) + output_data[0].shape[0], output_data[0].shape[1]))

    final_target = numpy.zeros(win_shift * (len(output_data) - 1) + output_data[0].shape[0])
    overlaping_label_count = numpy.zeros(final_output.shape)

    win_len = output_data[0].shape[0]
    tmp = numpy.ones(output_data[0].shape)

    # Loop on the overlaping windows
    for idx, tmp_t, tmp_d in zip(output_idx, output_target, output_data):
        start_idx = win_shift * idx
        stop_idx = start_idx + win_len

        overlaping_label_count[start_idx: stop_idx, :] += tmp
        final_output[start_idx: stop_idx, :] += tmp_d
        final_target[start_idx: stop_idx] += tmp_t

    # Divide by the number of overlapping values
    final_output /= overlaping_label_count
    final_target /= overlaping_label_count[:, 0].squeeze()

    return final_output, final_target


def _output2sns_byoffset(model_out, output_offset):
    """
    Asign the output of model with 2 values to calss label based on index
    :param output_offset: if is None the calss labels will be maximum of values,
    otherwise it return thethe index of maximum of class1 and (class2 + output_offset)
    """

    if output_offset == None:
        return numpy.argmax(model_out, axis=1)
    else:
        seq = numpy.zeros(len(model_out), dtype=int)
        for i in range(len(model_out)):
            if model_out[i, 1] + output_offset > model_out[i, 0]:
                seq[i] = 1
            else:
                seq[i] = 0
        return seq


class BLSTM(torch.nn.Module):
    """
    Bi LSTM model used for voice activity detection, speaker turn detection, overlap detection and resegmentation
    """
    def __init__(self,
                 input_size,
                 blstm_sizes,
                 num_layers):
        """

        :param input_size:
        :param blstm_sizes:
        """
        super(BLSTM, self).__init__()
        self.input_size = input_size
        self.blstm_sizes = blstm_sizes
        self.output_size = blstm_sizes * 2

        self.blstm_layers = torch.nn.LSTM(input_size,
                                          blstm_sizes,
                                          bidirectional=True,
                                          batch_first=True,
                                          num_layers=num_layers)

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        output, h = self.blstm_layers(inputs)
        return output

    def output_size(self):
        """

        :return:
        """
        return self.output_size


class SeqToSeq(torch.nn.Module):
    """
    Model used for voice activity detection or speaker turn detection
    This model can include a pre-processor to input raw waveform,
    a BLSTM module to process the sequence-to-sequence
    and other linear of convolutional layers
    """
    def __init__(self,
                 model_archi):

        super(SeqToSeq, self).__init__()

        # Load Yaml configuration
        with open(model_archi, 'r') as fh:
            cfg = yaml.load(fh, Loader=yaml.FullLoader)

        self.loss = cfg["loss"]
        self.feature_size = None

        """
        Prepare Preprocessor
        """
        self.preprocessor = None
        if "preprocessor" in cfg:
            if cfg['preprocessor']["type"] == "sincnet":
                self.preprocessor = SincNet(
                    waveform_normalize=cfg['preprocessor']["waveform_normalize"],
                    sample_rate=cfg['preprocessor']["sample_rate"],
                    min_low_hz=cfg['preprocessor']["min_low_hz"],
                    min_band_hz=cfg['preprocessor']["min_band_hz"],
                    out_channels=cfg['preprocessor']["out_channels"],
                    kernel_size=cfg['preprocessor']["kernel_size"],
                    stride=cfg['preprocessor']["stride"],
                    max_pool=cfg['preprocessor']["max_pool"],
                    instance_normalize=cfg['preprocessor']["instance_normalize"],
                    activation=cfg['preprocessor']["activation"],
                    dropout=cfg['preprocessor']["dropout"]
                )
                self.feature_size = self.preprocessor.dimension

        """
        Prepare sequence to sequence  network
        """
        # Get Feature size
        if self.feature_size is None:
            self.feature_size = cfg["feature_size"]

        input_size = self.feature_size
        sequence_to_sequence_layers = []
        for k in cfg["sequence_to_sequence"].keys():
            if k.startswith("blstm"):
                sequence_to_sequence_layers.append((k, BLSTM(input_size=input_size,
                                                             blstm_sizes=cfg["sequence_to_sequence"][k]["output_size"],
                                                             num_layers=cfg["sequence_to_sequence"][k]["num_layers"])))
                input_size = cfg["sequence_to_sequence"][k]["output_size"] * 2

        self.sequence_to_sequence = torch.nn.Sequential(OrderedDict(sequence_to_sequence_layers))

        """
        Prepare post-processing network
        """
        # Create sequential object for the second part of the network
        self.post_processing_activation = torch.nn.Tanh()
        post_processing_layers = []
        for k in cfg["post_processing"].keys():

            if k.startswith("lin"):
                post_processing_layers.append((k, torch.nn.Linear(input_size,
                                                                  cfg["post_processing"][k]["output"])))
                input_size = cfg["post_processing"][k]["output"]

            elif k.startswith("activation"):
                post_processing_layers.append((k, self.post_processing_activation))

            elif k.startswith('batch_norm'):
                post_processing_layers.append((k, torch.nn.BatchNorm1d(input_size)))

            elif k.startswith('dropout'):
                post_processing_layers.append((k, torch.nn.Dropout(p=cfg["post_processing"][k])))

        self.post_processing = torch.nn.Sequential(OrderedDict(post_processing_layers))
        self.post_processing.apply(init_weights)

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        if self.preprocessor is not None:
            x = self.preprocessor(inputs)
            x = x.permute(0, 2, 1)
        else:
            x = inputs
        x = self.sequence_to_sequence(x)
        x = self.post_processing(x)
        return x

    def predict(self,
                batch_size,
                validation_loader,
                device,
                shift, # former step
                output_rate,
                th_in, #output_offset=None,
                th_out,
                only_lab_generation=False):
        """
            A MODIFIER POU NE PRENDRE QUE LE NOM DU FICHIER WAV ET CRÉER LE DATZA LOADER À L'INTERIEUR

        :param model:
        :param validation_loader:
        :param device:
        :param only_lab_generation: only generating lab file for each wav file without evaluation
        :return:
        """

        recall = 0.0
        precision = 0.0
        accuracy = 0.0
        loss = 0.0

        output_data = []
        output_target = []
        output_idx = []

        sm = torch.nn.Softmax(dim=2)
        with torch.no_grad():

            for batch_idx, (win_idx, data, target) in tqdm.tqdm(enumerate(validation_loader)):
                target = target.squeeze().cpu().numpy()
                output = sm(self.forward(data.to(device))).cpu().numpy()

                for ii in range(output.shape[0]):
                    output_data.append(output[ii])
                    output_target.append(target[ii])
                    output_idx.append(int(win_idx[ii]))

        # Unfold outputs by averaging sliding windows
        final_output, final_target = multi_label_combination(output_idx,
                                                             output_target,
                                                             output_data,
                                                             shift,
                                                             output_rate,
                                                             mode="mean")

        vad = numpy.zeros(final_output.shape[0], dtype='bool')
        speech = False
        ii = 0
        while ii < final_output.shape[0]:
            if final_output[ii, 1] > th_in and not speech:
                speech = True
            elif final_output[ii, 1] < th_out and speech:
                speech = False
            vad[ii] = speech
            ii += 1

        #if not only_lab_generation:
        #    final_target = _unfold(final_target, target.cpu().numpy(), shift)

        #sad_seq = output2sns_byoffset(numpy.array(final_output), threshold)

        #if not only_lab_generation:
        #    recall, precision, accuracy, tp, fp, tn, fn = calc_recall(sad_seq, (numpy.array(final_target) != 0),
        #                                                              uemfile_dir=uemfile_dir)

        #    if precision != 0 or recall != 0:
        #        f_measure = 2 * (precision) * (recall) / (precision + recall)
        #        logging.critical(
        #            'Validation: [{}/{} ({:.0f}%)]  Accuracy: {:.3f}  ' \
        #            'Recall: {:.3f}  Precision: {:.3f} ' \
        #            'F-Measure: {:.3f}'.format(batch_idx + 1,
        #                                       validation_loader.__len__(),
        #                                       100. * batch_idx / validation_loader.__len__(),
        #                                       100.0 * accuracy,
        #                                       100.0 * recall,
        #                                       100.0 * precision,
        #                                       f_measure)
        #        )
        #    return 100.0 * accuracy, 100.0 * recall, 100.0 * precision, sad_seq, tp, fp, tn, fn
        #else:
        #    return 0, 0, 0, sad_seq, 0, 0, 0, 0
        return vad

def seqTrain(dataset_yaml,
             model_yaml,
             epochs=100,
             lr=0.0001,
             patience=10,
             model_name=None,
             tmp_model_name=None,
             best_model_name=None,
             multi_gpu=True,
             opt='sgd',
             log_interval=10,
             num_thread=10
             ):
    """

    :param data_dir:
    :param mode:
    :param duration:
    :param seg_shift:
    :param filter_type:
    :param collar_duration:
    :param framerate:
    :param epochs:
    :param lr:
    :param loss:
    :param patience:
    :param tmp_model_name:
    :param best_model_name:
    :param multi_gpu:
    :param opt:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start from scratch
    if model_name is None:
       model = SeqToSeq(model_yaml)
    # If we start from an existing model
    else:
        # Load the model
        logging.critical(f"*** Load model from = {model_name}")
        checkpoint = torch.load(model_name, map_location='cpu')
        model = SeqToSeq(model_yaml)
        model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1 and multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    else:
        print("Train on a single GPU")

    model.to(device)

    with open(dataset_yaml, "r") as fh:
        dataset_params = yaml.load(fh, Loader=yaml.FullLoader)
    """
    Create two dataloaders for training and evaluation
    """
    training_set, validation_set = create_train_val_seqtoseq(dataset_yaml)

    training_loader = DataLoader(training_set,
                                 batch_size=dataset_params["batch_size"],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=num_thread)

    validation_loader = DataLoader(validation_set,
                                   batch_size=dataset_params["batch_size"],
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=num_thread)

    """
    Set the training options
    """
    if opt == 'sgd':
        _optimizer = torch.optim.SGD
        _options = {'lr': lr, 'momentum': 0.9}
    elif opt == 'adam':
        _optimizer = torch.optim.Adam
        _options = {'lr': lr}
    elif opt == 'rmsprop':
        _optimizer = torch.optim.RMSprop
        _options = {'lr': lr}

    params = [
        {
            'params': [
                param for name, param in model.named_parameters() if 'bn' not in name
            ]
        },
        {
            'params': [
                param for name, param in model.named_parameters() if 'bn' in name
            ],
            'weight_decay': 0
        },
    ]

    optimizer = _optimizer([{'params': model.parameters()},], **_options)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    best_accuracy = 0.0
    best_accuracy_epoch = 1
    curr_patience = patience

    for epoch in range(1, epochs + 1):
        # Process one epoch and return the current model
        if curr_patience == 0:
            print(f"Stopping at epoch {epoch} for cause of patience")
            break
        model = train_epoch(model,
                            epoch,
                            training_loader,
                            optimizer,
                            log_interval,
                            device=device)

        # Cross validation here
        accuracy, val_loss = cross_validation(model, validation_loader, device=device)
        logging.critical("*** Cross validation accuracy = {} %".format(accuracy))

        # Decrease learning rate according to the scheduler policy
        scheduler.step(val_loss)
        print(f"Learning rate is {optimizer.param_groups[0]['lr']}")

        # remember best accuracy and save checkpoint
        is_best = accuracy > best_accuracy
        best_accuracy = max(accuracy, best_accuracy)

        if type(model) is SeqToSeq:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'scheduler': scheduler
            }, is_best, filename=tmp_model_name + ".pt", best_filename=best_model_name + '.pt')
        else:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'scheduler': scheduler
            }, is_best, filename=tmp_model_name + ".pt", best_filename=best_model_name + '.pt')

        if is_best:
            best_accuracy_epoch = epoch
            curr_patience = patience
        else:
            curr_patience -= 1

    logging.critical(f"Best accuracy {best_accuracy * 100.} obtained at epoch {best_accuracy_epoch}")


def train_epoch(model, epoch, training_loader, optimizer, log_interval, device):
    """

    :param model:
    :param epoch:
    :param training_loader:
    :param optimizer:
    :param log_interval:
    :param device:
    :param clipping:
    :return:
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.FloatTensor([0.9,0.1]).to(device))

    recall = 0.0
    precision = 0.0
    accuracy = 0.0

    for batch_idx, (data, target) in enumerate(training_loader):
        target = target.squeeze()
        optimizer.zero_grad()
        output = model(data.to(device))
        output = output.permute(1, 2, 0)
        target = target.permute(1, 0)

        loss = criterion(output, target.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()

        rc, pr, acc = calc_recall(output.data, target, device)
        recall += rc.item()
        precision += pr.item()
        accuracy += acc.item()

        if batch_idx % log_interval == 0:
            batch_size = target.shape[0]
            if precision!=0 or recall!=0:
                f_measure = 2 * (precision / ((batch_idx + 1))) * (recall / ((batch_idx+1))) /\
                            ((precision / ((batch_idx + 1) ))+(recall / ((batch_idx + 1))))
                logging.critical(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}  Accuracy: {:.3f}  '\
                        'Recall: {:.3f}  Precision: {:.3f}  '\
                        'F-Measure: {:.3f}'.format(epoch,
                                                  batch_idx + 1,
                                                  training_loader.__len__(),
                                                  100. * batch_idx / training_loader.__len__(), loss.item(),
                                                  100.0 * accuracy / ((batch_idx + 1)),
                                                  100.0 * recall/ ((batch_idx+1)),
                                                  100.0 * precision / ((batch_idx+1)),
                                                  f_measure)
                )
            else:
                print(f"precision = {precision} and recall = {recall}")

    return model


def cross_validation(model, validation_loader, device):
    """

    :param model:
    :param validation_loader:
    :param device:
    :return:
    """
    model.eval()

    recall = 0.0
    precision = 0.0
    accuracy = 0.0
    loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            target = target.squeeze()
            output = model(data.to(device))
            output = output.permute(1, 2, 0)
            target = target.permute(1, 0)

            loss += criterion(output, target.to(device))

            rc, pr, acc = calc_recall(output.data, target, device)
            recall += rc.item()
            precision += pr.item()
            accuracy += acc.item()

    batch_size = target.shape[0]
    if precision != 0 or recall != 0:
        f_measure = 2 * (precision / ((batch_idx + 1))) * (recall / ((batch_idx + 1))) / \
                    ((precision / ((batch_idx + 1))) + (recall / ((batch_idx + 1))))
        logging.critical(
            'Validation: [{}/{} ({:.0f}%)]  Loss: {:.6f}  Accuracy: {:.3f}  ' \
            'Recall: {:.3f}  Precision: {:.3f}  '\
            'F-Measure: {:.3f}'.format(batch_idx + 1,
                                      validation_loader.__len__(),
                                      100. * batch_idx / validation_loader.__len__(),
                                      loss.item() / (batch_idx + 1),
                                      100.0 * accuracy / ((batch_idx + 1)),
                                      100.0 * recall / ((batch_idx + 1)),
                                      100.0 * precision / ((batch_idx + 1)),
                                      f_measure)
        )

    return 100.0 * accuracy / ((batch_idx + 1)), loss/(batch_idx + 1)


def calc_recall(output,target,device):
    """

    :param output:
    :param target:
    :param device:
    :return:
    """
    y_trueb = target.to(device)
    y_predb = output
    rc = 0.0
    pr = 0.0
    acc= 0.0
    for b in range(y_trueb.shape[-1]):
        y_true = y_trueb[:,b]
        y_pred = y_predb[:,:,b]
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)

        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        epsilon = 1e-7

        pr+= tp / (tp + fp + epsilon)
        rc+= tp / (tp + fn + epsilon)
        a=(tp+tn)/(tp+fp+tn+fn+epsilon)

        acc+=(tp+tn)/(tp+fp+tn+fn+epsilon)
    rc/=len(y_trueb[0])
    pr/=len(y_trueb[0])
    acc/=len(y_trueb[0])

    return rc,pr,acc



