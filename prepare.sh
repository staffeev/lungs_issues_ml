flag=$1
pip install opendatasets
# git init  ### Я просто обычно уже репо через гит устанавливаю и поэтому это немного мешает. 
# git clone https://github.com/staffeev/lungs_issues_ml.git
# cd lungs_issues_ml
if [[ -z "$flag" ]]; then
    python3 scripts/creating_dataset.py --download
else
    python scripts/creating_dataset.py --download
fi
mkdir dataset/data/train_images_masked
python3 scripts/mask_images.py