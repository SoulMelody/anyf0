import csv
import functools
import math
import pathlib
from typing import BinaryIO

import click
import mido
import numpy as np
import librosa
import pretty_midi
from parselmouth import TextGrid
from parselmouth.praat import call
from tgt.core import IntervalTier

from anyf0.f0_extractor import F0Extractor
from anyf0.vshp import VocalShifterPatternType, VocalShifterProjectData

@click.group()
def cli():
    pass

@cli.command()
@click.argument('vshp_path', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('--method', type=click.Choice([
    'dio',
    'harvest',
    'pyin',
    'piptrack',
    'yin',
    'swipe',
    'salience',
    'swift',
    'praat_ac',
    'praat_cc'
]), default='praat_ac')
def getf0(vshp_path: pathlib.Path, method: str) -> None:
    vshp_data = VocalShifterProjectData.parse_file(vshp_path)
    pattern_indexes = []
    for i in range(vshp_data.project_metadata.pattern_count):
        if vshp_data.pattern_datas[i].header.pattern_type == VocalShifterPatternType.WAVE:
            wav_path = vshp_data.pattern_metadatas[i].path_and_ext.split(b'\x00')[0].decode('gbk')
            click.echo(f"Pattern {i}: {wav_path}")
            pattern_indexes.append(str(i))
    pattern_index = click.prompt(
        "Pattern Index",
        type=click.Choice(pattern_indexes),
        value_proc=int
    )
    wav_path = vshp_data.pattern_metadatas[pattern_index].path_and_ext.split(b'\x00')[0].decode('gbk')
    wav_path = pathlib.PureWindowsPath(wav_path)
    if wav_path.is_absolute():
        normlized_path = wav_path.as_posix()
    else:
        normlized_path = vshp_path.parent.joinpath(wav_path.as_posix())
    pattern_data = vshp_data.pattern_datas[pattern_index]
    sample_rate = pattern_data.header.sample_rate
    hop_length = sample_rate / pattern_data.header.frame_length

    f0_min = math.floor(librosa.midi_to_hz(pattern_data.header.key_min))
    f0_max = math.ceil(librosa.midi_to_hz(pattern_data.header.key_min + pattern_data.header.key_scale))
    click.echo(f"Path: {normlized_path}")
    click.echo(f"Sample Rate: {sample_rate}")
    click.echo(f"F0 Min: {f0_min}")
    click.echo(f"F0 Max: {f0_max}")

    f0_extractor = F0Extractor(
        normlized_path, sample_rate=sample_rate, hop_length=hop_length,
        f0_min=f0_min, f0_max=f0_max, method=method
    )
    f0 = f0_extractor.extract_f0()
    pattern_data.points = [
        {
            key: max(round(f0[i]), 0) if key == 'ori_pit' and i < len(f0) and not np.isnan(f0[i]) else value
                for key, value in pattern_data.points[i].items() if key != '_io'
        }
        for i in range(pattern_data.header.points_count)
    ]
    VocalShifterProjectData.build_file(vshp_data, vshp_path.with_suffix('.new.vshp'))
    click.echo("Done")


@cli.command()
@click.argument('midi_file', type=click.File(mode="rb"))
@click.argument('txt_path', type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path))
@click.option("--default-lyric", type=str, default="la")
@click.option("--charset", type=str, default="utf-8")
def export_lyric(midi_file: BinaryIO, txt_path: pathlib.Path, default_lyric: str = "la", charset: str = "utf-8") -> None:
    """
    Export lyric from midi to a text label file (which can be imported into a pattern later via vocalshifter)
    """
    mido.MidiFile = functools.partial(mido.MidiFile, charset=charset)
    midi_obj = pretty_midi.PrettyMIDI(midi_file)
    with txt_path.open("w", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["start", "end", "lyric"], dialect="excel-tab")
        time2lyrics = {
            lyric.time: lyric.text for lyric in midi_obj.lyrics
        }
        for i, track in enumerate(midi_obj.instruments):
            click.echo(f"track {i} has {len(track.notes)} notes.")
        midi_track_index = click.prompt(
            "MIDI Track Index",
            type=click.IntRange(0, len(midi_obj.instruments), max_open=True, clamp=True),
        )
        midi_track: pretty_midi.Instrument = midi_obj.instruments[midi_track_index]
        note: pretty_midi.Note
        for note in midi_track.notes:
            if len(time2lyrics):
                if note.start not in time2lyrics:
                    continue
                lyric = time2lyrics[note.start]
            else:
                lyric = default_lyric
            csv_writer.writerow(
                {
                    "start": note.start,
                    "end": note.end,
                    "lyric": lyric
                }
            )
    click.echo("Done")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.argument('txt_path', type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path))
def export_labels(input_path: pathlib.Path, txt_path: pathlib.Path) -> None:
    """
    Export lyric from Praat TextGrid to a text label file (which can be imported into a pattern later via vocalshifter)
    """
    datas = call("Read from file", str(input_path))
    tg_index = click.prompt(
        "TextGrid Index",
        type=click.Choice([
            str(i) for i, data in enumerate(datas)
            if isinstance(data, TextGrid)
        ]),
        value_proc=int
    )
    text_grid = datas[tg_index].to_tgt()
    name2tiers = {
        tier.name: tier for tier in text_grid.tiers
        if isinstance(tier, IntervalTier)
    }
    tier_name = click.prompt(
        "IntervalTier Index",
        type=click.Choice(list(name2tiers)),
    )
    tier = name2tiers[tier_name]
    with txt_path.open("w", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["start", "end", "lyric"], dialect="excel-tab")
        for interval in tier._objects:
            csv_writer.writerow(
                {
                    "start": interval.start_time,
                    "end": interval.end_time,
                    "lyric": interval.text
                }
            )
    click.echo("Done")

    


if __name__ == '__main__':
    cli()