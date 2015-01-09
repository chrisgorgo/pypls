## THIS IS A WORKING GUIDE TO SETUP A VIRTUAL ENVIROMENT IN ORDER TO RUN PLS.
## PLEASE TAKE INTO ACCOUNT THE VERSIONS OF THE USED MODULES

cd('=/.../<dir name>')
git clone https://github.com/chrisfilo/pypls.git
echo $PYTHONPATH
export PYTHONPATH=/.../dir/pypls/:$PYTHONPATH
git clone https://github.com/rasbt/pyprind.git
export PYTHONPATH=/.../dir/pypls/:/.../dir/pyprind/:$PYTHONPATH
git clone https://github.com/giampaolo/psutil.git
export PYTHONPATH=/.../dir/pypls/:/.../dir/psutil/:$PYTHONPATH
cd pyprind/
git tag
git checkout 2.3.0
virtualenv --help
virtualenv pls_env
. pls_env/bin/activate 

pip install -I numpy==1.7.0
pip install scipy
pip install -I scikit-learn==0.14

pip install psutil
pip install ipython
