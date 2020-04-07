# SDOBenchmark

## Installation

### General

Install Virtualenv

```
python3 -m pip install
```

Create venv subdirectory in project root

```
virtualenv -p python3 sdobenchmarkvenv
```

Activate the virtual Environment:

```
source sdobenchmarkvenv/bin/activate
```

Install the packages mentioned in the **requirements.txt**, with:

```
pip install -r requirements.txt
```

### Jupyter

Install Jupyter Lab

```
pip install jupyterlab
```

### Change Python Kernel

Install new kernel, to use virtualenv

``` bash
ipython kernel install --user list kernels
```

```
jupyter kernelspec list
```

*remove kernel (only if necessary)*

```bash
jupyter kernelspec uninstall unwanted-kernel
```

Add virtualenv path to ipython kernel

edit: `/home/<username>/.local/share/jupyter/kernels/<projectname>/kernel.json`

and add the path to the virtualenv python file

like that: 

```json
{
 "argv": [
  "/home/pirate/Desktop/SDOBenchmark/NickatFHNW/sdobenchmark/sdobenchmarkvenv/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "mnistvenv",
 "language": "python"
}
```

**IMPORTANT:** download ipykernel after entering the virtualenv

```bash
pip install ipykernel
```





## Training and Test Results



| DATASET                | IMG_SIZE | BATCH_SIZE | EPOCHS | TEST_AMOUNT | RESULT           |
| ---------------------- | -------- | ---------- | ------ | ----------- | ---------------- |
| 682xflare_682xno_flare | 50       | 100        | 1      | 0.1         | ~0.5             |
| 682xflare_682xno_flare | 50       | 100        | **5**  | 0.1         | **~0.69 - 0.79** |
| 682xflare_682xno_flare | 256      | 100        | 5      | 0.1         | ~0.515           |
| 682xflare_682xno_flare | 256      | 100        | 1      | 0.1         | ~0.515           |
| 682xflare_682xno_flare | 80       | 100        | 1      | 0.1         | ~0.522           |
| 682xflare_682xno_flare | 80       | 100        | 5      | 0.1         | ~0.706-0.757     |
|                        |          |            |        |             |                  |
|                        |          |            |        |             |                  |
|                        |          |            |        |             |                  |

