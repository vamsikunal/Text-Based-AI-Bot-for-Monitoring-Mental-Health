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
Copyright 2014-2021 Sylvain Meignier
"""

import pyaudio


class AudioPlayer:
    """
    Player implemented with PyAudio

    http://people.csail.mit.edu/hubert/pyaudio/

    Mac OS X:

      brew install portaudio
      pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz
    """
    def __init__(self, wav):
        """

        :param wav:
        """
        self.p = pyaudio.PyAudio()
        self.pos = 0
        self.stream = None
        self._open(wav)

    def callback(self, in_data, frame_count, time_info, status):
        """

        :param in_data:
        :param frame_count:
        :param time_info:
        :param status:
        :return:
        """
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        """

        :param wav:
        :return:
        """
        self.wf = wav
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(),
                rate = self.wf.getframerate(),
                output=True,
                stream_callback=self.callback)
        self.pause()

    def play(self):
        """

        :return:
        """
        self.stream.start_stream()

    def pause(self):
        """

        :return:
        """
        self.stream.stop_stream()

    def seek(self, seconds = 0.0):
        """

        :param seconds:
        :return:
        """
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        """

        :return:
        """
        return float(self.pos)/self.wf.getframerate()

    def playing(self):
        """

        :return:
        """
        return self.stream.is_active()

    def close(self):
        """

        :return:
        """
        self.stream.close()
        self.wf.close()
        self.p.terminate()

