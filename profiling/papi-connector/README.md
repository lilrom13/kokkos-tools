## Using the PAPI Connector

To use the PAPI connector you have to compile it against the latest PAPI version that includes the new high-level API.

Details about PAPI's new high-level API can be found here:
https://bitbucket.org/icl/papi/wiki/Home

#### Install PAPI (from version 6.0) or latest git version

```console
git clone https://bitbucket.org/icl/papi.git
```
Installation instructions can be found here:
https://icl.utk.edu/papi/software/index.html

#### Install the PAPI connector
```console
export PAPI_ROOT=<Path of PAPI>
make
```
This will generate a dynamic library `kp_papi_connector.so`.

#### Run kokkos application with PAPI recording enabled
Before executing the Kokkos application you have to set the environment variable `KOKKOS_PROFILE_LIBRARY` to point to the dynamic library. Also add the library path of PAPI and PAPI connector to `LD_LIBRARY_PATH`.

```console
export KOKKOS_PROFILE_LIBRARY=kp_papi_connector.so
export LD_LIBRARY_PATH=<PAPI-LIB-PATH>:<PAPI-CONNECTOR-LIB-PATH>:$LD_LIBRARY_PATH
```

As implemented in PAPI’s high-level API, default PAPI events are measured if you have not specified any specific events. You can specify arbitrary events using the environment variable `PAPI_EVENTS`.

```console
export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"
```

Note that all environment variables that are used by PAPI can be set as well, see PAPI's documentation.

#### Choosing what is measured

By default the connector opens a PAPI region around every Kokkos event: every kernel, deep copy, profile region and profile section. Reading counters that often can be expensive. To measure only some events, point `KOKKOS_TOOLS_PAPI_FILTER` to a filter file:

```console
export KOKKOS_TOOLS_PAPI_FILTER=filter.txt
```

```
# one rule per line: <type>: <pattern>
region: Total
region: TimeLoop
```

- Types: `region` (profile regions, matched by name), `region-path` (profile regions, matched by the `/`-joined names of the enclosing profile regions, outermost first, the region itself included, for example `Total/TimeLoop/Exchange`), `kernel` (`parallel_for`, `parallel_reduce` and `parallel_scan`), `copy` (deep copies, matched by the label of the destination View) and `section` (profile sections).
- A pattern is an ECMAScript regular expression matched against the whole name: `kernel: .*Flux.*` selects every kernel whose name contains `Flux`. Escape regular expression characters in literal names, for example `kernel: Kokkos::View::initialization \[rho\] via memset`.
- Everything after the first `:` is the pattern, so names containing `:` work (`region: AMR: adapt`). `#` starts a comment at the beginning of a line or after a space.
- With a filter file, only matching events are measured; a type without rules is never measured. Without `KOKKOS_TOOLS_PAPI_FILTER`, every event is measured, as before.
- A missing or invalid filter file stops the program at initialization, with the file and line of the error.

PAPI's high-level API supports at most 10 nested regions, and deactivates its output for the whole run beyond that. The connector therefore never opens more than 10 nested PAPI regions; deeper events are not measured and a warning is printed once.

#### Names in the output

| Kokkos event | Region name in the PAPI output |
|---|---|
| profile region `TimeLoop` | `kokkosp_profile_region:TimeLoop` |
| kernel `flux` | `kokkosp_parallel_for:flux` (also `kokkosp_parallel_reduce:`, `kokkosp_parallel_scan:`) |
| any deep copy | `kokkosp_deep_copy` |
| profile section `IO` with id 3 | `IO:3` |

#### Fencing

By default Kokkos fences the device before every kernel and profile region callback, so asynchronous device work is attributed to the right region. Set `KOKKOS_TOOLS_GLOBALFENCES=0` to turn these fences off: the overhead is lower, but device work may be counted in a later region.

#### Cost

Measured on an AMR hydrodynamics code with the CUDA backend and PAPI 7.3.0 (about 481,000 Kokkos events per run):

| Setup | Time loop |
|---|---|
| Without the connector | 17.8 s |
| Every event, CPU counters (`PAPI_TOT_INS,PAPI_TOT_CYC`) | 24.9 s |
| 6 selected regions, CPU counters | 17.8 s |
| Every event, NVML GPU energy (separate run, 12.7 s without the connector) | 859 s |

Measuring a few long regions is the way to read GPU energy counters, which take milliseconds to read and update only every 20 to 100 ms.

#### Analyze PAPI output
During the finalization phase of a Kokkos application, an output file is generated automatically. The output format is JSON and lists several PAPI events for each parallel construct or profile section. If not specified by the user, the output file will be located in the current working directory in the "papi" folder.

PAPI also offers a Python script which enhances the output
by creating some derived metrics, like IPC, MFlops/s, and MFlips/s as well as real and processor time in case the corresponding PAPI events have been recorded.

```console
papi_hl_output_writer.py --source papi
```
