SDO Benchmark
=============

Project Setup
-------------

1. Update the virtualenv module: `pip install -U virtualenv`
2. Create a virtual environment inside the root directory:
`virtualenv .venv`
3. Activate the newly created virtual environment from the command line:
`source .venv/bin/activate` on linux
or `.venv\Scripts\activate.bat` on Windows.
4. Install the requirements: `pip install -r requirements.txt`


Build Documentation
-------------------
In a terminal with an activate virtual environment, go to the `docs` directory.

To build a LaTeX PDF, run `make latexpdf`. To create a website run either
`make html` (for a full website) or `make singlehtml` (for a one-page documentation).

The resulting files are found inside the `_build` directory.


Example Usage
-------------

### Full Data Set using default parameters
In a terminal with an activate virtual environment, run

`python -m flares.load_data OUTPUT_DIRECTORY EMAIL_ADDRESS`

where **OUTPUT_DIRECTORY** is the path to the target directory and **EMAIL_ADDRESS**
a registered *JSOC* email address
(see the [JSOC website](http://jsoc.stanford.edu/ajax/register_email.html)).

### Benchmark
TODO


Known caveats
-------------

Solar cycle
Detector degradation