git clone --recursive https://github.com/princeton-vl/infinigen.git

conda init bash && . ~/.bashrc
cd infinigen
conda create --name infinigen python=3.10 -y
conda activate infinigen && bash install.sh
