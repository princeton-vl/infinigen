git clone --recursive git@github.com:princeton-vl/procgen.git

conda init bash && . ~/.bashrc
cd infinigen
conda create --name infinigen python=3.10 -y
conda activate infinigen && bash install.sh
