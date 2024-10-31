pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install xformers==0.0.22
pip install pyrender

cd submodules/Connected_components_PyTorch
python setup.py install

cd ../diff-gaussian-rasterization
python setup.py install

cd ../diff-gaussian-rasterization-personal
python setup.py install

cd ../simple-knn
python setup.py install

cd ../co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
mkdir -p checkpoints
cd checkpoints
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..

cd ../../model/curope
python setup.py install
cd ../..