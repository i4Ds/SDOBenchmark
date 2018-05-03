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


Example Usage
-------------

### Full Data Set using default parameters
In a terminal with an activate virtual environment, run

`python -m flares.load_data OUTPUT_DIRECTORY EMAIL_ADDRESS`

where **OUTPUT_DIRECTORY** is the path to the target directory and **EMAIL_ADDRESS**
a registered *JSOC* email address
(see the [JSOC website](http://jsoc.stanford.edu/ajax/register_email.html)).