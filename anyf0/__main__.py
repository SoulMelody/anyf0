import math
import pathlib

import click
import numpy as np
import librosa

from anyf0.f0_extractor import F0Extractor
from anyf0.vshp import VocalShifterPatternType, VocalShifterProjectData


@click.command()
@click.argument('vshp_path', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('--method', type=click.Choice(['torchcrepe', 'dio', 'harvest', 'yin', 'pyin', 'swipe', 'salience', 'parselmouth']), default='parselmouth')
def getf0(vshp_path: pathlib.Path, method: str) -> None:
    vshp_data = VocalShifterProjectData.parse_file(vshp_path)
    pattern_index = int(click.prompt(
        "Pattern Index",
        type=click.Choice([
            str(i) for i in range(vshp_data.project_metadata.pattern_count)
            if vshp_data.pattern_datas[i].header.pattern_type == VocalShifterPatternType.WAVE
        ])
    ))
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
        normlized_path, sampling_rate=sample_rate, hop_length=hop_length,
        f0_min=f0_min, f0_max=f0_max, method=method
    )
    f0 = f0_extractor.extract_f0()
    pattern_data.points = [
        {
            key: round(
                f0[i]) if key == 'ori_pit' and i < len(f0) and not np.isnan(f0[i]) else value
                for key, value in pattern_data.points[i].items() if key != '_io'
        }
        for i in range(pattern_data.header.points_count)
    ]
    VocalShifterProjectData.build_file(vshp_data, vshp_path.with_suffix('.new.vshp'))
    click.echo("Done")


if __name__ == '__main__':
    getf0()