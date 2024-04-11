flag=$1
pip install opendatasets
git init
git clone https://github.com/staffeev/lungs_issues_ml.git
cd lungs_issues_ml/
if [[ -z "$flag" ]]; then
    python3 scripts/creating_dataset.py --download
else
    python scripts/creating_dataset.py --download
fi
