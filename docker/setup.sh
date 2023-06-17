eval `ssh-agent -s` && ssh-add ~/.ssh/id_ed25519
git clone --recursive git@github.com:princeton-vl/procgen.git

conda init bash && . ~/.bashrc
cd procgen
conda create --name procgen python=3.10 -y
conda activate procgen && bash install.sh