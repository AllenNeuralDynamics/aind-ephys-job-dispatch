import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import numpy as np
from pathlib import Path
import json


# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se

from spikeinterface.core.core_tools import SIJsonEncoder


data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")


# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)

input_group = parser.add_mutually_exclusive_group()
input_help = "Which 'loader' to use (aind | spikeglx | nwb)"
input_group.add_argument("--input", default="aind", help=input_help, choices=["aind", "spikeglx", "nwb"])
input_group.add_argument("static_input", nargs="?", default="aind", help=input_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = True if args.static_concatenate and args.static_concatenate.lower() == "true" else args.concatenate
    INPUT = args.static_input or args.input

    print(f"Running job dispatcher with the following parameters:")
    print(f"\tCONCATENATE RECORDINGS: {CONCAT}")
    print(f"\tINPUT: {INPUT}")

    if INPUT == "aind":
        print("Parsing AIND input folder")
        # find ecephys sessions to process
        # for pipelines, the session data should to be mapped to the "data/ecephys_session" folder
        if (
            (data_folder / "ecephys").is_dir()
            or (data_folder / "ecephys_compressed").is_dir()
            or (data_folder / "ecephys_clipped").is_dir()
        ):
            ecephys_sessions = [data_folder]
        else:
            ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()]

        # not needed, we can parallelize
        # assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
        job_dict_list = []
        for session in ecephys_sessions:
            session_name = None
            if (session / "data_description.json").is_file():
                data_description = json.load(open(session / "data_description.json", "r"))
                session_name = data_description["name"]
            session_folder_path = session.relative_to(data_folder)

            ecephys_base_folder = session / "ecephys"
            compressed = False
            if (ecephys_base_folder / "ecephys_compressed").is_dir():
                # most recent folder organization
                compressed = True
                ecephys_compressed_folder = ecephys_base_folder / "ecephys_compressed"
                ecephys_folder = ecephys_base_folder / "ecephys_clipped"
            elif (session / "ecephys_compressed").is_dir():
                compressed = True
                ecephys_compressed_folder = session / "ecephys_compressed"
                ecephys_folder = session / "ecephys_clipped"
            else:
                # uncompressed data
                ecephys_folder = ecephys_base_folder

            print(
                f"Session: {session_name}")
            print(f"\tOpen Ephys folder: {str(ecephys_folder)}")
            if compressed:
                print(f"\tZarr compressed folder: {str(ecephys_compressed_folder)}")

            # get blocks/experiments and streams info
            num_blocks = se.get_neo_num_blocks("openephys", ecephys_folder)
            stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(ecephys_folder, block_index=0, stream_name=stream_names[0])
            record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
            experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
            exp_ids = list(experiments.keys())
            experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

            print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")
            print("\tRecording to be processed in parallel:")
            for block_index in range(num_blocks):
                for stream_name in stream_names:
                    # skip NIDAQ and NP1-LFP streams
                    if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "Rhythm" not in stream_name:
                        experiment_name = experiment_names[block_index]
                        exp_stream_name = f"{experiment_name}_{stream_name}"

                        if not compressed:
                            recording = se.read_openephys(ecephys_folder, stream_name=stream_name, block_index=block_index)
                        else:
                            recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

                        if CONCAT:
                            recordings = [recording]
                        else:
                            recordings = si.split_recording(recording)

                        HAS_CHANNEL_GROUPS = len(np.unique(recording.get_channel_groups())) > 1

                        for i_r, recording in enumerate(recordings):
                            if CONCAT:
                                recording_name = f"{exp_stream_name}_recording"
                            else:
                                recording_name = f"{exp_stream_name}_recording{i_r + 1}"

                            total_duration = np.round(recording.get_total_duration(), 2)

                            if HAS_CHANNEL_GROUPS:
                                for group_name, recording_group in recording.split_by("group").items():
                                    recording_name_group = f"{recording_name}_group{group_name}"
                                    print(f"\t\t{recording_name_group} - Duration: {total_duration} s - Num. channels: {recording_group.get_num_channels()}")
                                    job_dict = dict(
                                        session_name=session_name,
                                        recording_name=str(recording_name_group),
                                        recording_dict=recording_group.to_dict(
                                            recursive=True,
                                            relative_to=data_folder
                                        )
                                    )
                                    job_dict_list.append(job_dict)
                            else:
                                print(f"\t\t{recording_name} - Duration: {total_duration} s - Num. channels: {recording.get_num_channels()}")

                                job_dict = dict(
                                    session_name=session_name,
                                    recording_name=str(recording_name),
                                    recording_dict=recording.to_dict(
                                        recursive=True,
                                        relative_to=data_folder
                                    )
                                )
                                job_dict_list.append(job_dict)
    elif INPUT == "spikeglx":
        # get blocks/experiments and streams info
        spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        assert len(spikeglx_folders) == 1, "Attach one SpikeGLX folder at a time"
        spikeglx_folder = spikeglx_folders[0]
        session_name = spikeglx_folder.name
        num_blocks = se.get_neo_num_blocks("spikeglx", spikeglx_folder)
        stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)

        for block_index in range(num_blocks):
            for stream_name in stream_names:
                if "nidq" not in stream_name and "lf" not in stream_name:
                    recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name, block_index=block_index)
                    if CONCAT:
                        recordings = [recording]
                    else:
                        recordings = si.split_recording(recording)

                    HAS_CHANNEL_GROUPS = len(np.unique(recording.get_channel_groups())) > 1

                    for i_r, recording in enumerate(recordings_segments):
                        job_dict = dict(
                            session_name=session_name,
                        )
                        if CONCAT:
                            recording_name = f"block{block_index}_{stream_name}_recording"
                        else:
                            recording_name = f"block{block_index}_{stream_name}_recording{i_r + 1}"

                        total_duration = np.round(recording.get_total_duration(), 2)

                        if HAS_CHANNEL_GROUPS:
                            for group_name, recording_group in recording.split_by("group").items():
                                recording_name += f"_group{group_name}"
                                print(f"\t\t{recording_name} - Duration: {total_duration} s - Num. channels: {recording_group.get_num_channels()}")
                                job_dict["recording_name"] = recording_name
                                job_dict["recording_dict"] = recording_group.to_dict(
                                    recursive=True,
                                    relative_to=data_folder
                                )
                                job_dict_list.append(job_dict)
                        else:
                            print(f"\t\t{recording_name} - Duration: {total_duration} s - Num. channels: {recording.get_num_channels()}")

                            job_dict["recording_name"] = recording_name
                            job_dict["recording_dict"] = recording.to_dict(
                                recursive=True,
                                relative_to=data_folder
                            )
                            job_dict_list.append(job_dict)
    elif INPUT == "nwb":
        raise NotImplementedError

    for i, job_dict in enumerate(job_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    print(f"Generated {len(job_dict_list)} job config files")
