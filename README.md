# Dispatch jobs for AIND ephys pipeline
## aind-ephys-job-dispatch


### Description

This simple capsule is designed to dispatch jobs for the AIND pipeline. 

It assumes the data is stored in the `data/` directory, and creates as many JSON files 
as the number of jobs that can be run in parallel. Each job consists of a recording with spiking activity that needs spike sorting.

### Inputs

The `data/` folder must include an `ecephys` folder with a single recorded session.

The data ingestion layer supports three input formats:

- `aind`: data folder created at AIND (e.g., "ecephys_664438_2023-04-12_14-59-51") with the `ecephys` folder containing
  the `ecepys_compressed` and `ecephys_clipped` folders. 
- `spikeglx`: a SpikeGLX generated folder
- `nwb`: a single NWB file (either in HDF5 or Zarr backend)

The ecephys data can have multiple experiments (or blocks), each with multiple streams (probes) and continuous recordings (segments). Different recordings can be concatenated together to form a single recording (and spike sorted together) or treated as separate recordings (and spike sorted separately).

### Parameters

The `code/run` script takes 2 arguments:

```bash
  --concatenate         Whether to concatenate recordings (segments) or not. Default: False
  --input {aind,spikeglx,nwb}
                        Which 'loader' to use (aind | spikeglx | nwb)
```

### Output

The output of this capsule is a list of JSON files in the `results/` folder, containing the parameters for a spike sorting job. 

Each JSON file contains the following fields:

- `session_name`: the session name (e.g., "ecephys_664438_2023-04-12_14-59-51")
- `recording_name`: the recording name, which will correspond to output folders downstreams (e.g, "experiment1_Record Node 101#Neuropix-PXI-100.probeA-AP_recording1")
- `recording_dict`: the SpikeInterface dict representation of the recording with paths relative to the `data` folder