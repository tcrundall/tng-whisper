import sys
import traceback
import typing as typ
import numpy as np
import time

from gstreamer import GstContext, GstPipeline, GstApp, Gst

import gstreamer.utils as utils

MIC_PIPELINE = utils.to_gst_string(["pulsesrc",
                                   "audioconvert",
                                    "audio/x-raw, channels=1, format=S16LE, rate=48000",
                                    # "wavenc",
                                    # "oggmux",
                                    # "filesink location=bar.ogg"])
                                    "appsink emit-signals=True"])


def on_buffer(sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
    """Callback on 'new-sample' signal"""
    # Emit 'pull-sample' signal
    # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

    sample = sink.emit("pull-sample")  # Gst.Sample

    if isinstance(sample, Gst.Sample):
        array = extract_buffer(sample)
        print(
            "Received {type} with shape {shape} of type {dtype}".format(type=type(array),
                                                                        shape=array.shape,
                                                                        dtype=array.dtype))
        return Gst.FlowReturn.OK

    return Gst.FlowReturn.ERROR



def extract_buffer(sample: Gst.Sample) -> np.ndarray:
    """Extracts Gst.Buffer from Gst.Sample and converts to np.ndarray"""

    buffer = sample.get_buffer()  # Gst.Buffer

    print(buffer.pts, buffer.dts, buffer.offset)

    caps_format = sample.get_caps().get_structure(0)  # Gst.Structure

    # print("caps:" + caps_format.get_list())
    # print("caps:" + ' - '.join(dir(caps_format)))
    # print("caps format:" + caps_format.get_value('format'))

    # GstVideo.VideoFormat
    # video_format = GstVideo.VideoFormat.from_string(
    #     caps_format.get_value('format'))

    # audio_format = GstAudio.AudioFormat.from_string(caps_format.get_value('format'))

    # w, h = caps_format.get_value('width'), caps_format.get_value('height')
    # c = utils.get_num_channels(video_format)

    buffer_size = buffer.get_size()
    print(f"buffer size: {buffer_size}")
    # shape = (h, w, c) if (h * w * c == buffer_size) else buffer_size
    data = buffer.extract_dup(0, buffer_size)
    # array = np.ndarray(shape=(buffer_size), buffer=data,
    #                    dtype=utils.get_np_dtype(audio_format))
    array = np.ndarray(shape=(buffer_size // 2), buffer=data, dtype=np.int16)

    # TODO: FREE BUFFER!

    return np.squeeze(array)  # remove single dimension if exists


    # return np.zeros(109)
    # pass

def read_mic() -> None:
    """

    pass
    :return:
    """
    print("Hello!")
    with GstContext():
        with GstPipeline(MIC_PIPELINE) as pipeline:
            appsink = pipeline.get_by_cls(GstApp.AppSink)[0]
            appsink.connect("new-sample", on_buffer, None)

            # while not pipeline.is_done:
            cnt = 1
            while cnt < 2:
                cnt += 1
                time.sleep(1)
                print("first")
                time.sleep(1)
                print("second")
                time.sleep(1)
                print("third")

    print("Done")


