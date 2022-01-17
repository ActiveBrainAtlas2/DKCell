## CSHL cell detector setup

This python package runs on python2. Here are the necessary steps for installing
everything on Ubuntu 20.04 when you have no python2. If you already have python2
and virutalenv installed, look below.

1. Install python2.7: `sudo apt install python2`
1. Install dev packages: `sudo apt install python2-dev`
1. Get pip2: `wget https://bootstrap.pypa.io/pip/2.7/get-pip.py`
1. Install pip2 `sudo -H python2 get-pip.py`
1. Install virtualenv: `sudo -H /usr/local/bin/pip install virtualenv`
1. Create a virtualenv, you might need to do a chown on the directory first: 
`virtualenv /usr/local/share/cell_detector`
1. Activate venv: `source /usr/local/share/cell_detector/bin/activate`
1. Get DKCell: `git clone git@github.com:ActiveBrainAtlas2/DKCell.git`
1. Install requirements: `pip install -r requirements`
1. Run: `python test_frcnn_full_DK_May_2021_GPU0.py` and look for errors.

### For Ubuntu with python2 and virtualenv already installed (works on basalis)
1. Create a virtualenv: `virtualenv ~/.virtualenvs/cell_detector
1. Activate: `source ~/.virtualenvs/cell_detector/bin/activate
1. Get DKCell: `git clone git@github.com:ActiveBrainAtlas2/DKCell.git`
1. Install requirements: `pip install -r requirements`
1. Run: `python test_frcnn_full_DK_May_2021_GPU0.py` and look for errors.
