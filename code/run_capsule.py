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


MAX_NUM_NEGATIVE_TIMESTAMPS = 10
MAX_TIMESTAMPS_DEVIATION_MS = 1


data_folder = Path("../data")
results_folder = Path("../results")


# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)

split_group = parser.add_mutually_exclusive_group()
split_help = "Whether to process different groups separately"
split_group.add_argument("--split-groups", action="store_true", help=split_help)
split_group.add_argument("static_split_groups", nargs="?", default="false", help=split_help)

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", default="false", help=debug_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)

input_group = parser.add_mutually_exclusive_group()
input_help = "Which 'loader' to use (aind | spikeglx | nwb)"
input_group.add_argument("--input", default="aind", help=input_help, choices=["aind", "spikeglx", "openephys", "nwb"])
input_group.add_argument("static_input", nargs="?", help=input_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = True if args.static_concatenate and args.static_concatenate.lower() == "true" else args.concatenate
    SPLIT_GROUPS = (
        True if args.static_split_groups and args.static_split_groups.lower() == "true" else args.split_groups
    )
    DEBUG = args.debug or args.static_debug.lower() == "true"
    DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)
    INPUT = args.static_input or args.input

    print(f"Running job dispatcher with the following parameters:")
    print(f"\tCONCATENATE RECORDINGS: {CONCAT}")
    print(f"\tSPLIT GROUPS: {SPLIT_GROUPS}")
    print(f"\tDEBUG: {DEBUG}")
    print(f"\tDEBUG DURATION: {DEBUG_DURATION}")
    print(f"\tINPUT: {INPUT}")

    print(f"Parsing {INPUT} input folder")
    recording_dict = {}
    if INPUT == "aind":
        # find ecephys sessions to process
        #
        # - for pipelines, the session data should to be mapped to the "data/ecephys_session" folder
        # - for standalone capsule runs, the data is in "data/ecephys_{session_name}"
        ecephys_sessions = [
            p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()
        ]

        for session_folder in ecephys_sessions:
            session_name = None
            if (session_folder / "data_description.json").is_file():
                data_description = json.load(open(session_folder / "data_description.json", "r"))
                session_name = data_description["name"]

            # in the AIND pipeline, the session folder is mapped to
            ecephys_base_folder = session_folder / "ecephys"

            if (ecephys_base_folder / "ecephys_compressed").is_dir():
                new_format = True
                ecephys_folder = ecephys_base_folder
            else:
                new_format = False
                ecephys_folder = session_folder

            compressed = False
            if (ecephys_folder / "ecephys_compressed").is_dir():
                # most recent folder organization
                compressed = True
                ecephys_compressed_folder = ecephys_folder / "ecephys_compressed"
                ecephys_openephys_folder = ecephys_folder / "ecephys_clipped"
            else:
                # uncompressed data
                ecephys_openephys_folder = ecephys_base_folder

            print(f"\tSession name: {session_name}")
            print(f"\tOpen Ephys folder: {str(ecephys_openephys_folder)}")
            if compressed:
                print(f"\tZarr compressed folder: {str(ecephys_compressed_folder)}")

            # get blocks/experiments and streams info
            num_blocks = se.get_neo_num_blocks("openephysbinary", ecephys_openephys_folder)
            stream_names, stream_ids = se.get_neo_streams("openephysbinary", ecephys_openephys_folder)

            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(ecephys_openephys_folder, block_index=0, stream_name=stream_names[0])
            record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
            experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
            exp_ids = list(experiments.keys())
            experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

            print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")
            for block_index in range(num_blocks):
                for stream_name in stream_names:
                    # skip NIDAQ and NP1-LFP streams
                    if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "Rhythm" not in stream_name:
                        experiment_name = experiment_names[block_index]
                        exp_stream_name = f"{experiment_name}_{stream_name}"
                        if not compressed:
                            recording = se.read_openephys(
                                ecephys_openephys_folder, stream_name=stream_name, block_index=block_index
                            )
                        else:
                            recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")
                        recording_name = f"{exp_stream_name}_recording"
                        recording_dict[(session_name, recording_name)] = {}
                        recording_dict[(session_name, recording_name)]["raw"] = recording

                        # load the associated LF stream (if available)
                        if "AP" in stream_name:
                            stream_name_lf = stream_name.replace("AP", "LFP")
                            exp_stream_name_lf = exp_stream_name.replace("AP", "LFP")
                            try:
                                if not compressed:
                                    recording_lf = se.read_openephys(
                                        ecephys_openephys_folder, stream_name=stream_name_lf, block_index=block_index
                                    )
                                else:
                                    recording_lf = si.read_zarr(
                                        ecephys_compressed_folder / f"{exp_stream_name_lf}.zarr"
                                    )
                                recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                            except:
                                print(f"\t\tNo LFP stream found for {exp_stream_name}")

    elif INPUT == "spikeglx":
        # get blocks/experiments and streams info
        spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        assert len(spikeglx_folders) == 1, "The data folder should contain a single SpikeGLX folder at a time"
        spikeglx_folder = spikeglx_folders[0]
        session_name = spikeglx_folder.name
        stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)

        # spikeglx has only one block
        num_blocks = 1
        block_index = 0

        print(f"\tSession name: {session_name}")
        print(f"\tNum. streams: {len(stream_names)}")
        for stream_name in stream_names:
            if "nidq" not in stream_name and "lf" not in stream_name:
                recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)
                recording_name = f"block{block_index}_{stream_name}_recording"
                recording_dict[(session_name, recording_name)] = {}
                recording_dict[(session_name, recording_name)]["raw"] = recording

                # load the associated LF stream (if available)
                if "ap" in stream_name:
                    stream_name_lf = stream_name.replace("ap", "lf")
                    try:
                        recording_lf = se.read_spikeglx(spikeglx_folder, stream_name=stream_name_lf)
                        recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                    except:
                        print(f"\t\tNo LFP stream found for {stream_name}")

    elif INPUT == "openephys":
        # get blocks/experiments and streams info
        openephys_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        assert len(openephys_folders) == 1, "The data folder should contain a single OpenEphys folder at a time"
        openephys_folder = openephys_folders[0]
        session_name = openephys_folder.name
        num_blocks = se.get_neo_num_blocks("openephysbinary", openephys_folder)
        stream_names, stream_ids = se.get_neo_streams("openephysbinary", openephys_folder)

        print(f"\tSession name: {session_name}")
        print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")

        # load first stream to map block_indices to experiment_names
        rec_test = se.read_openephys(openephys_folder, block_index=0, stream_name=stream_names[0])
        record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
        experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
        exp_ids = list(experiments.keys())
        experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

        for block_index in range(num_blocks):
            for stream_name in stream_names:
                if "NI-DAQ" not in stream_name and "LFP" not in stream_name:
                    experiment_name = experiment_names[block_index]
                    exp_stream_name = f"{experiment_name}_{stream_name}"
                    recording = se.read_openephys(
                        openephys_folder, load_sync_timestamps=True, stream_name=stream_name, block_index=block_index
                    )
                    recording_name = f"{exp_stream_name}_recording"
                    recording_dict[(session_name, recording_name)] = {}
                    recording_dict[(session_name, recording_name)]["raw"] = recording

                    # load the associated LFP stream (if available)
                    if "AP" in stream_name:
                        stream_name_lf = stream_name.replace("AP", "LFP")
                        try:
                            recording_lf = se.read_openephys(
                                openephys_folder, stream_name=stream_name_lf, block_index=block_index
                            )
                            recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                        except:
                            print(f"\t\tNo LFP stream found for {stream_name}")

    elif INPUT == "nwb":
        # get blocks/experiments and streams info
        all_input_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        if len(all_input_folders) == 1:
            nwb_files = [p for p in all_input_folders[0].iterdir() if p.name.endswith(".nwb")]
        else:
            nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb")]
        print(f"nwb_files: {nwb_files}")
        if len(nwb_files) == 0:
            raise ValueError("No NWB files found in the data folder")
        elif len(nwb_files) > 1:
            raise ValueError("Multiple NWB files found in the data folder. Please only add one at a time")
        nwb_file = nwb_files[0]
        session_name = nwb_file.name

        # spikeglx has only one block
        num_blocks = 1
        block_index = 0

        electrical_series_paths = se.NwbRecordingExtractor.fetch_available_electrical_series_paths(nwb_file)

        print(f"\tSession name: {session_name}")
        print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(electrical_series_paths)}")
        for electrical_series_path in electrical_series_paths:
            # only use paths in acquisition
            if "acquisition" in electrical_series_path:
                stream_name = electrical_series_path.replace("/", "-")
                recording = se.read_nwb_recording(nwb_file, electrical_series_path=electrical_series_path)
                if recording.sampling_frequency < 10000:
                    print(
                        f"\t\t{electrical_series_path} is probably an LFP signal (sampling frequency: "
                        f"{recording.sampling_frequency} Hz). Skipping"
                    )
                    continue
                recording_name = f"block{block_index}_{stream_name}_recording"
                recording_dict[(session_name, recording_name)] = {}
                recording_dict[(session_name, recording_name)]["raw"] = recording

    # populate job dict list
    job_dict_list = []
    print("Recording to be processed in parallel:")
    for session_recording_name in recording_dict:
        session_name, recording_name = session_recording_name
        recording = recording_dict[session_recording_name]["raw"]
        recording_lfp = recording_dict[session_recording_name].get("lfp", None)

        HAS_LFP = recording_lfp is not None
        if CONCAT:
            recordings = [recording]
            recordings_lfp = [recording_lfp] if HAS_LFP else None
        else:
            recordings = si.split_recording(recording)
            recordings_lfp = si.split_recording(recording_lfp) if HAS_LFP else None

        for recording_index, recording in enumerate(recordings):
            if not CONCAT:
                recording_name_segment = f"{recording_name}{recording_index + 1}"
            else:
                recording_name_segment = f"{recording_name}"

            if HAS_LFP:
                recording_lfp = recordings_lfp[recording_index]

            # timestamps should be monotonically increasing, but we allow for small glitches
            skip_times = False
            for segment_index in range(recording.get_num_segments()):
                times = recording.get_times(segment_index=segment_index)
                times_diff = np.diff(times)
                num_negative_times = np.sum(times_diff < 0)

                if num_negative_times > 0:
                    print(f"\t\t{recording_name} - Times not monotonically increasing.")
                    if num_negative_times > MAX_NUM_NEGATIVE_TIMESTAMPS:
                        print(f"\t\t{recording_name} - Skipping timestamps for too many negative timestamps")
                        skip_times = True
                        break
                    if np.max(np.abs(times_diff)) * 1000 > MAX_TIMESTAMPS_DEVIATION_MS:
                        print(
                            f"\t\t{recording_name} - Skipping timesstamps for too large deviation ({np.max(np.abs(times_diff))} ms)"
                        )
                        skip_times = True
                        break

            if skip_times:
                recording.reset_times()

            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0, end_frame=int(DEBUG_DURATION * recording.sampling_frequency)
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)
                if HAS_LFP:
                    recording_lfp_list = []
                    for segment_index in range(recording_lfp.get_num_segments()):
                        recording_lfp_one = si.split_recording(recording_lfp)[segment_index]
                        recording_lfp_one = recording_lfp_one.frame_slice(
                            start_frame=0, end_frame=int(DEBUG_DURATION * recording_lfp.sampling_frequency)
                        )
                        recording_lfp_list.append(recording_lfp_one)
                    recording_lfp = si.append_recordings(recording_lfp_list)

            duration = np.round(recording.get_total_duration(), 2)

            # if multiple channel groups, process in parallel
            if SPLIT_GROUPS and len(np.unique(recording.get_channel_groups())) > 1:
                for group_name, recording_group in recording.split_by("group").items():
                    recording_name_group = f"{recording_name_segment}_group{group_name}"
                    job_dict = dict(
                        session_name=session_name,
                        recording_name=str(recording_name_group),
                        recording_dict=recording_group.to_dict(recursive=True, relative_to=data_folder),
                        skip_times=skip_times,
                        duration=duration,
                        debug=DEBUG,
                    )
                    rec_str = f"\t{recording_name_group} - Duration: {duration} s - Num. channels: {recording_group.get_num_channels()}"
                    if HAS_LFP:
                        recording_lfp_group = recording_lfp.split_by("group")[group_name]
                        job_dict["recording_lfp_dict"] = recording_lfp_group.to_dict(
                            recursive=True, relative_to=data_folder
                        )
                        rec_str += f" (with LFP stream)"
                    print(rec_str)
                    job_dict_list.append(job_dict)
            else:
                job_dict = dict(
                    session_name=session_name,
                    recording_name=str(recording_name_segment),
                    recording_dict=recording.to_dict(recursive=True, relative_to=data_folder),
                    skip_times=skip_times,
                    duration=duration,
                    debug=DEBUG,
                )
                rec_str = f"\t{recording_name_segment} - Duration: {duration} s - Num. channels: {recording.get_num_channels()}"
                if HAS_LFP:
                    job_dict["recording_lfp_dict"] = recording_lfp.to_dict(recursive=True, relative_to=data_folder)
                    rec_str += f" (with LFP stream)"
                print(rec_str)
                job_dict_list.append(job_dict)

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    for i, job_dict in enumerate(job_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    print(f"Generated {len(job_dict_list)} job config files")
