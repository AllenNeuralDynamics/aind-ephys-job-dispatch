# Dispatch jobs for AIND ephys pipeline
## aind-ephys-job-dispatch


### Description

This simple capsule is designed to dispatch jobs for the AIND pipeline. 

It assumes the data is stored in the `data/` directory, and creates as many JSON files 
as the number of jobs that can be run in parallel. Each job consists of a recording with spiking activity that needs spike sorting.

### Inputs

The `data/` folder must include a single recorded session (e.g., "ecephys_664438_2023-04-12_14-59-51") with the `ecephys` (uncompressed Open Ephys output) or the `ecepys_compressed` and `ecephys_clipped` folders (processed with [aind-data-transfer](https://github.com/AllenNeuralDynamics/aind-data-transfer)). 
The ecephys data can have multiple experiments (or blocks), each with multiple streams (probes) and continuous recordings (segments).
Different recordings can be concatenated together to form a single recording (and spike sorted together) or treated as separate recordings (and spike sorted separately).


### Parameters

The `code/run` script takes 1 argument:

- `concat`: `false` (default) | `true`. If `true`, the capsule will concatenate all recordings together. If `false`, each recording will be spike sorted separately.


### Output

The output of this capsule is a list of JSON files in the `results/` folder, containing the parameters for a spike sorting job. 

Each JSON file contains the following fields:

- `experiment_name`: the Open Ephys experiment name (e.g., "experiment1")
- `block_index`: the corresponding NEO block index (e.g., 0)
- `stream_name`: the NEO/SpikeInterface stream name (e.g., "Record Node 101#Neuropix-PXI-100.probeA-AP")
- `session_name`: the session name (e.g., "ecephys_664438_2023-04-12_14-59-51")
- `session_folder_path`: the session folder path relative to data
- `segment_index`: the segment index (e.g., 0)
- `recording_name`: the recording name, which will correspond to output folders downstreams (e.g, "experiment1_Record Node 101#Neuropix-PXI-100.probeA-AP_recording1")